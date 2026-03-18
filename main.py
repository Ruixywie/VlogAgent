"""VlogAgent 入口脚本"""

import argparse
import logging
import sys

from src.agent import VlogAgent


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(description="VlogAgent - AI 视频美化系统")
    parser.add_argument("video", help="输入视频路径")
    parser.add_argument(
        "-c", "--config",
        default="configs/default.yaml",
        help="配置文件路径 (默认: configs/default.yaml)",
    )
    parser.add_argument(
        "-o", "--output",
        default="output",
        help="输出目录 (默认: output)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细日志输出",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    agent = VlogAgent(config_path=args.config)
    result = agent.run(args.video, output_dir=args.output)
    print(f"\n输出视频: {result}")


if __name__ == "__main__":
    main()
