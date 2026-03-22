"""从 Pexels 自动下载训练视频素材

使用方法：
  1. 去 https://www.pexels.com/api/ 注册获取免费 API Key
  2. 运行：python training/download_pexels.py --api-key YOUR_KEY
  3. 视频保存到 training/data/pexels_videos/
"""

import argparse
import os
import time
import json
import requests

# 搜索关键词 + 每个关键词下载数量
SEARCH_QUERIES = [
    ("nature landscape", 12),
    ("mountain scenery", 8),
    ("ocean waves beach", 8),
    ("city walk street", 10),
    ("urban skyline", 5),
    ("night city lights", 5),
    ("people walking", 8),
    ("cooking food kitchen", 8),
    ("restaurant cafe", 5),
    ("travel road trip", 10),
    ("train window view", 5),
    ("forest trees", 8),
    ("sunset sunrise", 8),
    ("indoor home living room", 5),
    ("garden flowers", 5),
]
# 总计约 120 段

API_BASE = "https://api.pexels.com/videos/search"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "pexels_videos")


def search_videos(api_key: str, query: str, count: int) -> list[dict]:
    """搜索视频，返回视频信息列表"""
    headers = {"Authorization": api_key}
    params = {
        "query": query,
        "per_page": min(count, 80),
        "page": 1,
        "orientation": "landscape",  # 横屏优先
    }

    resp = requests.get(API_BASE, headers=headers, params=params)
    if resp.status_code != 200:
        print(f"  搜索失败 [{resp.status_code}]: {resp.text[:200]}")
        return []

    data = resp.json()
    return data.get("videos", [])[:count]


def select_best_file(video_files: list[dict]) -> dict | None:
    """选择最合适的视频文件（优先 1080p，其次 720p）"""
    # 按高度排序，找最接近 1080 的
    candidates = sorted(video_files, key=lambda f: abs(f.get("height", 0) - 1080))
    for f in candidates:
        h = f.get("height", 0)
        if 720 <= h <= 1080:
            return f
    # 回退到任何可用的
    return candidates[0] if candidates else None


def download_video(url: str, save_path: str) -> bool:
    """下载单个视频"""
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))

        with open(save_path, "wb") as f:
            downloaded = 0
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)

        size_mb = os.path.getsize(save_path) / 1024 / 1024
        return True
    except Exception as e:
        print(f"  下载失败: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False


def main():
    parser = argparse.ArgumentParser(description="从 Pexels 下载训练视频")
    parser.add_argument("--api-key", required=True, help="Pexels API Key")
    parser.add_argument("--output", default=OUTPUT_DIR, help="输出目录")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 记录已下载的视频 ID（避免重复）
    manifest_path = os.path.join(args.output, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    else:
        manifest = {"downloaded": [], "total": 0}

    downloaded_ids = set(v["id"] for v in manifest["downloaded"])

    total_downloaded = 0
    total_skipped = 0

    for query, count in SEARCH_QUERIES:
        print(f"\n搜索: \"{query}\" (目标 {count} 段)")

        videos = search_videos(args.api_key, query, count)
        print(f"  找到 {len(videos)} 个结果")

        for video in videos:
            vid = video["id"]
            duration = video.get("duration", 0)

            # 跳过已下载的
            if vid in downloaded_ids:
                total_skipped += 1
                continue

            # 跳过太长或太短的
            if duration < 5 or duration > 60:
                continue

            # 选择合适的文件
            best_file = select_best_file(video.get("video_files", []))
            if not best_file:
                continue

            url = best_file["link"]
            width = best_file.get("width", 0)
            height = best_file.get("height", 0)
            filename = f"pexels_{vid}_{width}x{height}.mp4"
            save_path = os.path.join(args.output, filename)

            print(f"  下载 #{vid} ({duration}s, {width}x{height})... ", end="", flush=True)
            if download_video(url, save_path):
                size_mb = os.path.getsize(save_path) / 1024 / 1024
                print(f"OK ({size_mb:.1f}MB)")
                manifest["downloaded"].append({
                    "id": vid,
                    "query": query,
                    "duration": duration,
                    "width": width,
                    "height": height,
                    "filename": filename,
                })
                downloaded_ids.add(vid)
                total_downloaded += 1
            else:
                print("FAILED")

            # 控制请求频率（每小时 200 次限制）
            time.sleep(1)

        # 搜索请求之间间隔
        time.sleep(2)

    # 保存清单
    manifest["total"] = len(manifest["downloaded"])
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n完成！本次下载 {total_downloaded} 段，跳过 {total_skipped} 段")
    print(f"累计 {manifest['total']} 段视频，保存在 {args.output}")
    print(f"清单文件: {manifest_path}")


if __name__ == "__main__":
    main()
