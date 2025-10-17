"""全局日志管理"""

from typing import Optional
from loguru import logger

import os
from datetime import datetime


def _setup_logger(level: Optional[str] = None) -> None:
    logger.remove()
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=log_level,
        colorize=True,
        diagnose=False,
        backtrace=False,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
    )

    # 日期分层日志路径，例如 logs/2025/10/17.log
    today = datetime.now()
    datedir = os.path.join("logs", f"{today.year:04d}", f"{today.month:02d}")
    os.makedirs(datedir, exist_ok=True)
    logfile = os.path.join(datedir, f"{today.day:02d}.log")

    logger.add(
        logfile,
        level=log_level,
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
        enqueue=True,
        backtrace=False,
        diagnose=False,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    )


# 初始化全局日志器
_setup_logger()

# 对外暴露统一名称
log = logger

__all__ = ["log", "logger"]


if __name__ == "__main__":
    # 测试命令，根目录路径运行：uv run python -m src.knowledge_qa.log_manager
    log.info("测试日志")
    log.warning("测试警告")
    log.error("测试错误")
    log.critical("测试严重错误")
    log.debug("测试调试信息")
    log.trace("测试追踪信息")
    log.success("测试成功")
    log.info("测试日志")
    log.warning("测试警告")
    log.error("测试错误")
    log.critical("测试严重错误")