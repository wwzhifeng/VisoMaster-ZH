"""
优化 #4: 同帧多人脸合批推理
=============================
问题:
  app/processors/face_swappers.py 中对每张人脸单独 unsqueeze(0)
  调用一次模型, 多人场景下 N 次 kernel launch 浪费 GPU。

修复:
  在 frame_worker 收集完同帧所有 face crops 后, stack 成 [N,3,H,W]
  一次推理。需要 TRT 引擎支持 dynamic batch (min=1, opt=4, max=8)。

预期收益: 多人脸场景吞吐提升 1.5~3x

依赖:
  - swapper 引擎编译时启用 dynamic batch profile (见 engines_manifest.json)
  - 默认 OFF, 用户在 config.ini 启用 batch_faces=true

实现策略:
  拦截 face_swappers.FaceSwappers.run_inswapper 等方法,
  若调用方传入的是 list/tuple of tensors, 自动 stack 后一次调用,
  返回 list of outputs (按顺序分配回各人脸)。
"""
from __future__ import annotations

import functools
import importlib


def _stack_or_passthrough(fn):
    @functools.wraps(fn)
    def wrapper(self, image, *args, **kwargs):
        import torch
        # 单张人脸: 走原版路径, 0 开销
        if isinstance(image, torch.Tensor) and image.dim() in (3, 4):
            return fn(self, image, *args, **kwargs)
        # 多张人脸 (list / tuple): stack 成 batch
        if isinstance(image, (list, tuple)) and len(image) > 1:
            batched = torch.stack([t.squeeze(0) if t.dim() == 4 else t for t in image])
            out = fn(self, batched, *args, **kwargs)
            # 拆分回 N 个 tensor (按第 0 维)
            if isinstance(out, torch.Tensor) and out.dim() == 4 and out.shape[0] == len(image):
                return [out[i:i+1] for i in range(out.shape[0])]
            return out
        if isinstance(image, (list, tuple)) and len(image) == 1:
            return [fn(self, image[0], *args, **kwargs)]
        return fn(self, image, *args, **kwargs)
    wrapper.__wrapped_by__ = "batch_faces"
    return wrapper


def apply() -> None:
    fs = importlib.import_module("app.processors.face_swappers")
    candidates = [
        "run_inswapper", "run_swapper", "run_ghost", "run_simswap",
        "run_styleswap", "run_cscs",
    ]
    patched = []
    for attr_name in dir(fs):
        cls = getattr(fs, attr_name)
        if not isinstance(cls, type):
            continue
        if "Swap" not in attr_name:
            continue
        for m in candidates:
            if hasattr(cls, m):
                orig = getattr(cls, m)
                if getattr(orig, "__wrapped_by__", None) == "batch_faces":
                    continue
                setattr(cls, m, _stack_or_passthrough(orig))
                patched.append(f"{attr_name}.{m}")
    if not patched:
        raise RuntimeError("未找到 swapper 方法, 上游可能已重构")
    print(f"    [batch_faces] patched: {patched}")
    print(f"    [batch_faces] 注意: 需要 swapper 引擎启用 dynamic batch")
