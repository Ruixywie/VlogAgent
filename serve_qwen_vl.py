#!/usr/bin/env python3
"""Qwen2.5-VL-72B OpenAI 兼容 API 服务器（同步版）

使用方法：
    python3 serve_qwen_vl.py --model-path /data/models/Qwen2.5-VL-72B-Instruct --port 8000
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 全局模型
model = None
processor = None
MODEL_NAME = "qwen2.5-vl"


def load_model(model_path, device="0,1,2,3,4,5,6,7"):
    global model, processor

    # 必须在 import torch 之前设置设备（和 QwenVL.py 参考脚本一致）
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = device

    import torch
    try:
        import torch_npu
        from torch_npu.contrib import transfer_to_npu
        IS_ASCEND = True
        logger.info("昇腾 NPU 已加载")
    except ImportError:
        IS_ASCEND = False
        logger.info("未检测到 NPU，使用 CPU/GPU")

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    logger.info(f"加载模型: {model_path}")
    logger.info(f"可用设备: ASCEND={IS_ASCEND}, torch.npu.is_available()={torch.npu.is_available() if IS_ASCEND else 'N/A'}")
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    elapsed = time.time() - t0
    logger.info(f"模型加载完成: {elapsed:.1f}s")

    # 打印模型实际分布的设备
    if hasattr(model, 'hf_device_map'):
        devices_used = set(str(v) for v in model.hf_device_map.values())
        logger.info(f"模型分布设备: {devices_used}")
        logger.info(f"设备映射条目数: {len(model.hf_device_map)}")
    else:
        logger.info(f"模型设备: {next(model.parameters()).device}")


def do_inference(messages, max_tokens=2048, temperature=0.7):
    """同步推理"""
    from qwen_vl_utils import process_vision_info

    # 转换消息格式
    qwen_msgs = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            qwen_msgs.append({"role": msg["role"], "content": [{"type": "text", "text": content}]})
        elif isinstance(content, list):
            qwen_content = []
            for item in content:
                if item.get("type") == "text":
                    qwen_content.append({"type": "text", "text": item["text"]})
                elif item.get("type") == "image_url":
                    url = item["image_url"]["url"]
                    qwen_content.append({"type": "image", "image": url})
                elif item.get("type") == "video":
                    qwen_content.append({"type": "video", "video": item.get("video", "")})
            qwen_msgs.append({"role": msg["role"], "content": qwen_content})
        else:
            qwen_msgs.append({"role": msg["role"], "content": [{"type": "text", "text": str(content)}]})

    logger.info("构建输入...")
    text = processor.apply_chat_template(qwen_msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(qwen_msgs)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    logger.info(f"输入 token 数: {inputs.input_ids.shape}")
    target_device = "npu:0" if 'torch_npu' in sys.modules else model.device
    inputs = inputs.to(target_device)
    logger.info(f"输入设备: {inputs.input_ids.device}")

    import torch
    t0 = time.time()
    logger.info("开始生成...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 0.01),
            do_sample=temperature > 0.01,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    elapsed = time.time() - t0
    prompt_tokens = len(inputs.input_ids[0])
    completion_tokens = len(generated_ids_trimmed[0])
    logger.info(f"推理完成: {len(output_text)} 字, {elapsed:.1f}s, tokens={prompt_tokens}+{completion_tokens}")

    return output_text, prompt_tokens, completion_tokens


class APIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        logger.info(f"{self.client_address[0]} - {format % args}")

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health" or self.path == "/v1/health":
            self._send_json({"status": "ok", "model_loaded": model is not None})
        elif self.path == "/v1/models":
            self._send_json({
                "object": "list",
                "data": [{"id": MODEL_NAME, "object": "model", "created": int(time.time()), "owned_by": "local"}],
            })
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._send_json({"error": "not found"}, 404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            request = json.loads(body)

            messages = request.get("messages", [])
            max_tokens = request.get("max_tokens", 2048)
            temperature = request.get("temperature", 0.7)

            logger.info(f"收到请求: {len(messages)} 条消息, max_tokens={max_tokens}")

            output_text, prompt_tokens, completion_tokens = do_inference(
                messages, max_tokens, temperature
            )

            response = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": output_text},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            self._send_json(response)

        except Exception as e:
            logger.error(f"推理失败: {e}", exc_info=True)
            self._send_json({"error": {"message": str(e), "type": "server_error"}}, 500)


def run_server(host, port):
    server = HTTPServer((host, port), APIHandler)
    logger.info(f"API 服务启动: http://{host}:{port}")
    logger.info("按 Ctrl+C 停止")

    def shutdown(sig, frame):
        logger.info("正在关闭...")
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/data/models/Qwen2.5-VL-72B-Instruct")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()

    load_model(args.model_path, args.device)
    run_server(args.host, args.port)
