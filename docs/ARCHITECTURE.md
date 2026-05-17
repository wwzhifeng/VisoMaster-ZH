# 架构 + TRT 加速路径说明

理解本整合包的设计取舍。

---

## 整体架构

```
┌─────────────────────────────────────────────┐
│  用户双击 Start.bat                          │
└─────────────┬───────────────────────────────┘
              │ 拼接 PATH (DLL/ffmpeg) + 启动 pythonw
              ▼
┌─────────────────────────────────────────────┐
│  启动器 GUI (_internal/launcher/)            │
│  ┌─────────────────────────────────────┐   │
│  │ 状态卡片 + 操作按钮 + 日志区        │   │
│  │ - 检测环境 / 下载模型 / 编译引擎     │   │
│  │ - 预热引擎 / 启动主程序              │   │
│  └─────────────────────────────────────┘   │
└─────────────┬───────────────────────────────┘
              │ subprocess 启动 bootstrap.py
              ▼
┌─────────────────────────────────────────────┐
│  bootstrap.py                                │
│  - os.add_dll_directory (CUDA/cuDNN/TRT)    │
│  - PATH 添加 ffmpeg/bin                      │
│  - 创建 model_assets 软链接                  │
│  - 应用 monkey-patch (_internal/optimizations/) │
│  - runpy 执行上游 main.py                   │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│  上游 VisoMaster main.py                     │
│  (PySide6 主窗口 + ModelsProcessor)          │
└─────────────────────────────────────────────┘
```

---

## TRT 加速的两条路径（**关键**）

VisoMaster 对 TRT 的使用是**双轨制**，整合包必须理解这点：

### 路径 A：LivePortrait → TensorRTPredictor 直接加载

```
LivePortrait 6 个模型
  ↓ ONNX (用户下载)
  ↓ 启动器 [编译 LP 引擎] 用 EngineBuilder 编译
  ↓ engines/<GPU>_TRT<ver>_fp16/*.engine
  ↓ 主程序 ModelsProcessor.load_model_trt()
  ↓ TensorRTPredictor.deserialize_cuda_engine()
  ↓ 直接推理
```

- 走 `models_trt[]` 字典
- 启动器编译的引擎**直接被主程序使用**
- 用 `predict_async()` API，跟 ORT 接口不同

### 路径 B：其它所有模型 → ORT TensorRT EP

```
其它 40+ 模型 (检测/换脸/修复/超分/上色)
  ↓ ONNX
  ↓ 主程序 ORT InferenceSession(providers=['TensorrtExecutionProvider', ...])
  ↓ ORT 内部首次推理时编译 ONNX -> .engine
  ↓ 缓存到 tensorrt-engines/{ORT-hash}.engine
  ↓ 后续推理直接 load cache
```

- 走 `models[]` 字典
- ORT 用自己的 hash 命名缓存文件
- **启动器编译的引擎对这条路无用**（hash 对不上）
- 用 `session.run_with_iobinding()` API

### 为什么不能让启动器预编译路径 B 的引擎

ORT TensorRT EP 的 cache key 是：
```
sha256(onnx_bytes + EP_options + GPU_name + driver_version + TRT_version)
```

我们无法在不真正调用 ORT 的情况下复现这个 hash，所以**无法预先生成 ORT 认识的 .engine**。

唯一的预编译方式是**真的让 ORT 跑一次推理**。这需要：
- 实例化完整的 ModelsProcessor
- 给它 fake 输入（图片/视频帧）
- 调用 face_detectors / face_swappers / face_restorers 等真实方法
- 让 ORT 触发编译并落盘

实施起来很复杂（依赖 PySide6 主窗口 + 控制信号），benchmark.py 尝试过失败。**现状是不做**，用户首次跑视频时由 ORT 自己编译，卡 1~3 分钟，之后就快了。

---

## 关于运行时优化框架

`_internal/optimizations/` 下的 monkey-patch 是 **试验性** 的，**默认全部关闭**。
它们试图在上游代码外做注入优化：

| 模块 | 设想 | 现实 |
|------|------|------|
| `sync_optimizer` | LivePortrait pipeline 末尾单次同步 | 找不到上游入口方法，已禁用 |
| `fp16_io` | FP16 ONNX 模型用 FP16 IO binding | 上游 io_binding 调用签名多变，wrapper 不安全，已 no-op |
| `trt_extended` | 把 Face swapper/restorer 也走 TRT | 与上游 io_binding 接口冲突，已禁用 |
| `batch_faces` | 同帧多人脸合批推理 | 需要 dynamic batch 引擎 + 上游支持 list 输入，验证未通过 |
| `pipeline_threading` | GPU 单线程化 + CPU 双缓冲 | 最小实现（强制 num_threads=1），未深度重构 |
| `cuda_graph` | CUDA Graph 加速 LivePortrait | 占位 |

**结论**：这套框架目前没有可见收益。保留代码是为了：
1. 留作上游重构后再次尝试的脚手架
2. 给社区贡献者参考

需要试验时在 `config.ini [advanced]` 段对应键改 `true`。

---

## 引擎编译策略

`_internal/build_engines.py` 的工作流：

```
对每个 ONNX:
  1. 读 ONNX 输入 shape, 判断是否含动态维度 (-1)
  2. 是动态 -> 自建 builder + 启发式 profile
       超分类 (esrgan/bsrgan/ultra*) max=(1,3,1024,1024) 避免显存爆
       检测类 (det/scrfd) max=(1,3,1280,1280)
       其它 batch 动态 max=(8,3,H,W)
  3. 是静态 -> 自建 builder 不带 profile
  4. warping_spade 类自动加载 grid_sample_3d_plugin.dll
  5. 自建失败 -> fallback 到上游 EngineBuilder
  6. timing.cache 复用, 同 GPU 后续编译加速
```

引擎产物按 `engines/{GPU_NAME}_TRT{version}_fp16/` 隔离，
便于多卡用户和 TRT 升级场景。

---

## 模型路径兼容

上游用扁平结构 `./model_assets/foo.onnx`，整合包保持一致。
唯一例外：LivePortrait 模型在 `./model_assets/liveportrait_onnx/foo.onnx` 子目录下。

`bootstrap.py` 会在启动时把 `model_assets/` 目录用 directory junction 链接到 `models/`（如果存在），
方便老用户从 `models/` 目录迁移过来。
