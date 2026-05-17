"""
优化 #6 (实验性): CUDA Graph 加速 LivePortrait
================================================
问题:
  LivePortrait 6 个模型 shape 全固定, 每次推理重复 6 次 kernel launch
  + ctype 转换 + Python 调用栈开销。

修复:
  首次推理时 capture 整个 pipeline 为 CUDA Graph,
  后续直接 replay, 跳过所有 launch overhead。

预期收益: LivePortrait 延迟再降 10~15%

依赖:
  - 所有模型用同一个 cuda stream
  - 输入 / 输出 buffer 地址固定 (复用 pre-allocated tensors)
  - PyTorch >= 1.10 (我们用 2.7, OK)
  - TRT 引擎本身要支持 stream capture (10.7 OK)

风险:
  - 实现复杂, 容易翻车
  - 任何动态分支会导致 graph 失效
  - 默认 OFF, 待充分测试后再开启
"""
from __future__ import annotations


def apply() -> None:
    # 实验性占位 - 完整实现需要深度访问 LivePortrait 各模型的
    # TensorRTPredictor 实例和 stream / buffer 管理。
    # 这里仅打印提示, 真正实现待 LivePortrait 重构完成。
    print("    [cuda_graph] 实验性优化, 当前为占位实现")
    print("    [cuda_graph] 完整实现需深度访问 TensorRTPredictor 内部状态")
