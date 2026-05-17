# 更新日志

本整合包基于上游 [VisoMaster](https://github.com/visomaster/VisoMaster) 制作，
仅记录**整合包侧的改动**。上游核心功能更新见上游 CHANGELOG。

---

## [v2.0.0] - 2026-05-17

### 新增

- **图形化启动器**（PySide6）：状态卡片 + 操作按钮 + 实时日志，替代命令行
- **嵌入式 Python 运行时**：解压即用，不依赖系统 conda/CUDA/cuDNN
- **中文汉化**：上游纯英文 UI 全部翻译为中文
- **模型管理器**：交互式下载菜单 + 断点续传 + SHA256 校验 + 多镜像 fallback (hf-mirror/huggingface/github)
- **TRT 引擎编译器**：
  - 自动检测动态 shape 并生成 optimization profile
  - 自动加载 grid_sample_3d_plugin.dll
  - 上游 EngineBuilder + 自建 builder 双路径，互为 fallback
  - 按 GPU 型号 + TRT 版本隔离引擎目录，避免冲突
- **环境诊断工具** (`Check_GPU.bat` / `diagnose.bat`)
- **嵌入式 ffmpeg**：自动加入 PATH，避免视频读写崩溃
- **诊断模式**：闪退时前台日志，定位问题

### 适配

- **RTX 50xx (Blackwell sm_120)**：升级 TensorRT 10.7 → 10.16，原生支持
- **CUDA 12.8 + cuDNN 9.5 + PyTorch 2.7+cu128 + ONNX Runtime 1.21**
- 兼容 RTX 20xx ~ RTX 50xx 全系（cu128 包含 sm_70 到 sm_120 内核）

### 修复（绕过上游问题）

- 上游 EngineBuilder 在某些 dynamic shape 模型上崩溃（IndexError），改走自建 builder
- 上游 GridSample 5D 算子在 TRT 10.x 不支持，用 warping_spade-fix 替代
- 上游某些 ONNX 模型 max profile 设太大导致显存溢出，超分类自动降到 (1, 3, 1024, 1024)

### 已知限制

- 启动器编译的 42 个非 LP 引擎**不会被上游 ORT TensorRT EP 复用**（ORT 用自己的 hash 命名）。
  仅 LivePortrait 6 个引擎被上游 TensorRTPredictor 直接加载。
- 用户首次在主程序选 "TensorRT" provider 跑视频，第一次会卡 1~3 分钟（ORT 现场编译），这是上游设计，整合包无法消除。

### 优化框架（试验性，默认关闭）

`_internal/optimizations/` 下的运行时 monkey-patch，可在 `config.ini [advanced]` 段开关。
当前都是**试验性质**，未验证有实质收益：

- `sync_optimizer` - LivePortrait pipeline 末尾单次同步
- `fp16_io` - FP16 IO 绑定（已无操作）
- `trt_extended` - Face swapper/restorer 注入 TRT 引擎（与上游 io_binding 冲突，已禁用）
- `batch_faces` - 多人脸合批
- `pipeline_threading` - GPU 单线程化
- `cuda_graph` - CUDA Graph（占位）

---

## 未来计划

- [ ] DeepSeek 汉化的严格规则化（避免误改 dict key）
- [ ] headless 预编译脚本（让 ORT 自动编译所有用到的模型）
- [ ] 主程序窗口加 "首次编译" 进度提示，避免误以为崩溃
- [ ] 给上游提 PR：dynamic shape profile 自动化、Blackwell 适配
