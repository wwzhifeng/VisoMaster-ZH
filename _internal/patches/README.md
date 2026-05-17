# Patches / Optimizations 说明

本整合包**不采用传统 `.patch` 文件**，所有优化通过 `_internal/optimizations/` 下的
运行时 monkey-patch 实现。

## 为什么不用 .patch 文件

- 上游代码更新会导致行号失效，patch 反复维护成本高
- monkey-patch 按方法名注入，对 refactor 更鲁棒
- 每项优化可通过 `config.ini [advanced]` 段独立开关，便于 A/B 测试
- 失败时只 warn 不中断，自动 fallback 到原版行为

## 优化项总览

| 优化项 | 模块 | 配置键 | 默认 | 预期收益 |
|--------|------|--------|------|---------|
| LivePortrait 单次同步 | `sync_optimizer.py` | `batch_synchronize` | ✓ | LP 延迟 -10~20% |
| FP16 IO 绑定 | `fp16_io.py` | `fp16_io_binding` | ✓ | 修复链路 +5~10% |
| Swapper/Restorer 走 TRT | `trt_extended.py` | `trt_swapper` | ✓ | 主链路 +30~60% |
| 多人脸合批 | `batch_faces.py` | `batch_faces` | · | 多脸场景 1.5~3x |
| GPU 单线程化 | `pipeline_threading.py` | `decouple_threading` | · | 稳定性提升 |
| CUDA Graph | `cuda_graph.py` | `enable_cuda_graph` | · | LP 额外 -10~15% (实验) |

## 启用方式

编辑 `config.ini` 的 `[advanced]` 段，把对应键设为 `true` 或 `false`。

## 验证优化是否生效

```
Start.bat
# 启动日志会打印:
# [bootstrap] 应用运行时优化:
#   [opt][ ok ] sync_optimizer       LivePortrait pipeline 末尾单次同步
#   [opt][ ok ] fp16_io              FP16 ONNX 模型用 FP16 IO binding
#   [opt][ ok ] trt_extended         Face swapper/restorer 走 TRT 引擎
#   [opt][skip] batch_faces          ...
```

## 如何添加新的优化项

1. 在 `_internal/optimizations/` 下创建新模块，导出 `apply()` 函数
2. 在 `__init__.py` 的 `OPTIMIZATIONS` 列表中注册 `(模块名, 配置键, 默认值, 描述)`
3. 在 `config.ini` 的 `[advanced]` 段加对应键
4. 在本文档表格中记录

## 兼容性提示

`trt_extended` 依赖 `engines/` 下存在对应 `.engine` 文件，未编译时自动跳过不影响主程序。

`batch_faces` 需要 swapper 引擎在编译时启用 dynamic batch profile
（见 `_internal/build_engines.py` 的 `profiles` 字段），否则会触发 TRT 报错回退到 batch=1。
