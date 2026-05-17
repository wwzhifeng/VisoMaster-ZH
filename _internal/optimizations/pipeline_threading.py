"""
优化 #5: GPU 推理单线程 + CPU 双缓冲
=====================================
问题:
  app/processors/video_processor.py 用每帧一个 FrameWorker 线程并发推理,
  但 GPU 上推理本身串行执行 + Python GIL, 多线程反而增加同步开销。

修复:
  - 引入推理线程 (1 个) 专门处理 GPU 工作
  - 预处理 (resize/affine) 在 CPU 多线程进行, 通过队列送入推理线程
  - 后处理 (compose/encode) 也在 CPU 多线程
  - 三段流水线解耦: [CPU prep N线程] -> [GPU 1线程] -> [CPU post M线程]

预期收益: 帧率稳定性提升, CPU 占用降低, 不一定提升峰值速度

实现策略:
  这是较侵入性的优化, 涉及 VideoProcessor 的 worker 调度重构。
  这里只做最小化干预: 把 num_threads 强制限制为 GPU 推理 1 线程,
  CPU 预后处理由 ORT 内部 intra/inter op 线程池处理。
  完整的三段流水线需要重写 worker 调度, 默认禁用。
"""
from __future__ import annotations

import importlib


def apply() -> None:
    vp = importlib.import_module("app.processors.video_processor")
    cls = None
    for n in dir(vp):
        obj = getattr(vp, n)
        if isinstance(obj, type) and "VideoProcessor" in n:
            cls = obj
            break
    if cls is None:
        raise RuntimeError("未找到 VideoProcessor 类")

    # 最小化干预: 拦截 __init__, 把 num_threads 限制为 1 (GPU 推理)
    orig_init = cls.__init__

    import functools
    @functools.wraps(orig_init)
    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        # 限制 GPU 推理并发到 1, 避免 CUDA context 争用
        if hasattr(self, "num_threads"):
            self._original_num_threads = self.num_threads
            self.num_threads = 1

    patched_init.__wrapped_by__ = "pipeline_threading"
    cls.__init__ = patched_init
    print("    [pipeline_threading] GPU 推理线程限制为 1 (CPU 处理交给 ORT 内部线程池)")
    print("    [pipeline_threading] 完整三段流水线重构未启用 (需深度改造)")
