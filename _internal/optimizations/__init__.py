"""
VisoMaster TRT Portable - 运行时优化框架
========================================
设计原则:
  1. 不修改上游源码, 全部通过 monkey-patch 注入
  2. 每个优化项独立模块, 可单独启用/禁用 (config.ini)
  3. patch 失败时只 warn, 不中断主程序 (fallback 到原版行为)
  4. 所有 patch 必须在 main.py 导入业务代码之前完成

启用方式: bootstrap.py 中调用 apply_all()
配置开关: config.ini [advanced] 段
"""
from __future__ import annotations

import importlib
import sys
import traceback
from configparser import ConfigParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "config.ini"

# 优化项注册表: (模块名, config_key, 默认启用, 描述)
OPTIMIZATIONS = [
    ("sync_optimizer",     "batch_synchronize",  False, "LivePortrait pipeline 末尾单次同步"),
    ("fp16_io",            "fp16_io_binding",    True,  "FP16 ONNX 模型用 FP16 IO binding"),
    ("trt_extended",       "trt_swapper",        False, "Face swapper/restorer 走 TRT 引擎 (与上游 io_binding 冲突, 暂禁用)"),
    ("batch_faces",        "batch_faces",        False, "同帧多人脸合批 (需 dynamic batch 引擎)"),
    ("pipeline_threading", "decouple_threading", False, "GPU 推理单线程 + CPU 双缓冲"),
    ("cuda_graph",         "enable_cuda_graph",  False, "CUDA Graph 加速 LivePortrait (实验)"),
]


def _load_cfg() -> ConfigParser:
    cfg = ConfigParser()
    if CONFIG_PATH.exists():
        cfg.read(CONFIG_PATH, encoding="utf-8")
    return cfg


def _is_enabled(cfg: ConfigParser, key: str, default: bool) -> bool:
    return cfg.getboolean("advanced", key, fallback=default)


def apply_all(verbose: bool = True) -> dict:
    """
    遍历优化注册表, 按 config.ini 开关激活。
    返回 {name: True/False/error_str}, 用于诊断。
    """
    cfg = _load_cfg()
    results = {}
    for name, key, default, desc in OPTIMIZATIONS:
        enabled = _is_enabled(cfg, key, default)
        if not enabled:
            results[name] = False
            if verbose:
                print(f"  [opt][skip] {name:<20} {desc}")
            continue
        try:
            mod = importlib.import_module(f"_internal.optimizations.{name}")
            if not hasattr(mod, "apply"):
                results[name] = "no apply()"
                continue
            mod.apply()
            results[name] = True
            if verbose:
                print(f"  [opt][ ok ] {name:<20} {desc}")
        except Exception as e:
            results[name] = f"FAIL: {type(e).__name__}: {e}"
            if verbose:
                # 只打一行简短错误, 不再 dump 整段 traceback
                print(f"  [opt][FAIL] {name:<20} {type(e).__name__}: {e}")
    return results


def status() -> None:
    cfg = _load_cfg()
    print(f"\n  优化项配置状态 ({CONFIG_PATH}):")
    for name, key, default, desc in OPTIMIZATIONS:
        en = _is_enabled(cfg, key, default)
        mark = "✓" if en else "·"
        print(f"    {mark} {name:<22} [{key}] {desc}")
