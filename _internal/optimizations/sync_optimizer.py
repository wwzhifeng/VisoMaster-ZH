"""
优化 #1: LivePortrait pipeline 末尾单次 synchronize
=====================================================
问题:
  app/processors/face_editors.py 中 LivePortrait 的 6 个 TRT 模型
  每次 predict_async 后立刻 torch.cuda.synchronize(), async 完全白给。

修复:
  - 拦截 LivePortrait 主流程入口
  - 内部 6 次 predict_async 都跳过 sync (替换为 no-op)
  - pipeline 结束时单次 torch.cuda.synchronize()
  - 同步开销从 6N 次降到 N 次

预期收益: LivePortrait 延迟降低 10~20%

实现策略:
  在 face_editors.LivePortrait 类的入口方法 (通常叫 run / process /
  generate_*) 外层套 wrapper, wrapper 内 monkeypatch
  torch.cuda.synchronize 为 no-op, 退出时恢复并显式同步一次。
"""
from __future__ import annotations

import functools
import importlib


# 用 nullcontext 替代 sync 调用
def _noop_sync():
    pass


def _wrap_pipeline_method(orig):
    """把一个方法包成 '内部禁 sync, 出口统一 sync' 模式"""
    @functools.wraps(orig)
    def wrapper(self, *args, **kwargs):
        import torch
        real_sync = torch.cuda.synchronize
        torch.cuda.synchronize = _noop_sync  # type: ignore
        try:
            return orig(self, *args, **kwargs)
        finally:
            torch.cuda.synchronize = real_sync  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    wrapper.__wrapped_by__ = "sync_optimizer"
    return wrapper


def apply() -> None:
    # face_editors 模块是 LivePortrait 主入口
    fe = importlib.import_module("app.processors.face_editors")

    # 上游可能的入口方法名 (尽量全覆盖, 找到哪个 patch 哪个)
    candidate_methods = [
        "run_liveportrait",
        "run_live_portrait",
        "process_liveportrait",
        "generate_liveportrait",
        "apply_face_editor",
        "run",
    ]

    # 上游 LivePortrait 既可能是顶层函数也可能是类方法
    patched = []

    # 1. 顶层函数
    for name in candidate_methods:
        if hasattr(fe, name) and callable(getattr(fe, name)):
            fn = getattr(fe, name)
            if getattr(fn, "__wrapped_by__", None) == "sync_optimizer":
                continue
            setattr(fe, name, _wrap_pipeline_method(fn))
            patched.append(f"fn:{name}")

    # 2. 类方法 (查找类名包含 Editor / LivePortrait 的)
    for attr_name in dir(fe):
        attr = getattr(fe, attr_name)
        if not isinstance(attr, type):
            continue
        if not any(k in attr_name for k in ("Editor", "LivePortrait", "FaceEditor")):
            continue
        for m in candidate_methods:
            if hasattr(attr, m):
                orig = getattr(attr, m)
                if getattr(orig, "__wrapped_by__", None) == "sync_optimizer":
                    continue
                setattr(attr, m, _wrap_pipeline_method(orig))
                patched.append(f"{attr_name}.{m}")

    if not patched:
        # 没找到入口不算错误, 上游可能 LivePortrait 入口名变了, 静默跳过
        print("    [sync_optimizer] 未找到匹配的 LivePortrait 入口方法, 跳过 (非致命)")
        return
    print(f"    [sync_optimizer] patched: {', '.join(patched)}")
