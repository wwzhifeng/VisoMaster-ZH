# VisoMaster-ZH

> 基于 [VisoMaster](https://github.com/visomaster/VisoMaster) 的中文整合包，针对 **RTX 50xx (Blackwell)** 适配，
> 提供图形化启动器、中文界面、嵌入式 Python 环境、自动模型管理与 TRT 引擎编译。

**性质**：本仓库是 **构建脚本 + 启动器**，不含上游 VisoMaster 源码和模型权重。
核心换脸/编辑/表情驱动功能来自 [VisoMaster](https://github.com/visomaster/VisoMaster) (GPL-3.0)。

---

## ✨ 这个整合包解决了什么

| 痛点 | 整合包做了什么 |
|------|---------------|
| 命令行操作，小白看不懂 | PySide6 图形化启动器 |
| 纯英文界面 | 中文汉化 |
| 要装 conda + CUDA + cuDNN | 嵌入式 Python 解压即用 |
| **RTX 50xx 跑不起来（TRT 10.7 不支持 sm_120）** | **升级到 TRT 10.16+，原生支持 Blackwell** |
| 模型下载经常超时/中断 | 多镜像 fallback + 断点续传 |
| 视频读写崩溃（缺 ffmpeg） | 自带 ffmpeg 自动配置 |
| 引擎换显卡冲突 | 按 GPU 型号隔离目录 |
| 出问题不知道怎么查 | 内置环境诊断 + 详细日志 |

> **注意**：本整合包**不提升上游推理性能**，主要价值在「移植适配 + 易用性」。
> 真正的 TRT 加速由上游 ORT TensorRT EP 提供，跟官方版速度一致。

---

## 🚀 用户使用：下载现成整合包

代码仓库不包含主包（几个 GB）和模型（几十 GB），去下方地址下载：

> **整合包下载**：[https://wangzhifeng.vip/](https://wangzhifeng.vip/)
> **GitHub Releases**：[https://github.com/wwzhifeng/VisoMaster-ZH/releases](https://github.com/wwzhifeng/VisoMaster-ZH/releases)

详细使用说明在主包内的 `README.txt`。

---

## 🛠 开发者：从源码构建整合包

如果你想自己 build 整合包（修改启动器/优化/汉化等）：

### 环境要求

- Windows 10 / 11
- Git
- 7-Zip (`winget install 7zip.7zip`)
- NVIDIA GPU（开发不需要，最终测试需要）
- 网络畅通（首次构建下载 ~8GB 依赖，建议挂代理）

### 一键构建

```powershell
git clone https://github.com/wwzhifeng/VisoMaster-ZH.git
cd VisoMaster-ZH
powershell -ExecutionPolicy Bypass -File .\_internal\_build_package.ps1 -Stage all
```

`-Stage all` 会按顺序：
1. 下载嵌入式 Python 3.10.11
2. pip 装 torch 2.7+cu128 / TRT 10.16+ / ORT 1.21 等约 8GB 依赖
3. git clone 上游 VisoMaster 源码到 `app/`
4. 校验所有脚本/配置存在
5. 7z 打包到 `dist/`

### 分阶段构建

```powershell
.\_internal\_build_package.ps1 -Stage python   # 仅 Python 运行时
.\_internal\_build_package.ps1 -Stage deps     # 仅 pip 依赖
.\_internal\_build_package.ps1 -Stage source   # 仅上游源码
.\_internal\_build_package.ps1 -Stage pack     # 仅打包
```

详见 [docs/BUILD.md](docs/BUILD.md)。

---

## 🖼️ 启动器界面

```
┌─ VisoMaster 中文整合包 启动器 ──────────────────┐
│ ┌─运行环境──┐ ┌─模型状态─┐ ┌─TRT 引擎─┐       │
│ │ RTX 5070Ti│ │   54/57  │ │   6/6   │       │
│ │ TRT 10.16 │ │ 已下载    │ │ LP 已备  │       │
│ └──────────┘ └─────────┘ └─────────┘       │
│                                                │
│  📥 下载模型   ⚙️ 编译 LP 引擎   🔥 预热 LP    │
│  🔍 重检环境    🛠 高级设置      📂 打开目录    │
│  🧹 清理缓存                                  │
│                                                │
│  ╔════════ 启 动 主 程 序 ════════╗            │
│                                                │
│  日志输出区...                                 │
└────────────────────────────────────────────────┘
```

---

## 📁 仓库结构

```
VisoMaster-ZH/
├── _internal/                  # 整合包内部脚本（本仓库核心）
│   ├── bootstrap.py            # 启动引导：DLL 注入 + ffmpeg PATH
│   ├── check_env.py            # 环境诊断
│   ├── model_manager.py        # 模型下载管理
│   ├── build_engines.py        # TRT 引擎编译器
│   ├── manifest.json           # 模型清单（URL/SHA256）
│   ├── requirements_portable_cu128.txt
│   ├── _build_package.ps1      # 一键打包脚本
│   ├── _gen_manifest.py        # 从上游 models_data.py 生成清单
│   ├── _benchmark.py           # 性能 benchmark
│   ├── diagnose.bat            # 诊断模式（前台日志）
│   ├── launcher/               # PySide6 启动器 GUI
│   │   ├── main_window.py
│   │   ├── workers.py
│   │   └── dialogs.py
│   └── optimizations/          # 运行时优化框架（monkey-patch）
│       ├── __init__.py
│       ├── sync_optimizer.py
│       ├── fp16_io.py
│       ├── trt_extended.py
│       ├── batch_faces.py
│       ├── pipeline_threading.py
│       └── cuda_graph.py
├── Start.bat                   # 用户入口（双击启动）
├── config.ini                  # 用户配置示例
├── README.md                   # 本文档
├── LICENSE                     # GPL-3.0
├── CHANGELOG.md                # 改动记录
└── docs/
    ├── BUILD.md                # 开发者构建详解
    └── ARCHITECTURE.md         # 架构 + TRT 加速路径说明
```

---

## 🤝 致谢

- [VisoMaster](https://github.com/visomaster/VisoMaster) — 上游核心项目，GPL-3.0
- [FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait) — engine_builder 思路来源
- 所有为 VisoMaster 贡献模型的研究者

---

## 📜 许可证

本仓库所有脚本采用 **GPL-3.0**（与上游一致），详见 [LICENSE](LICENSE)。

按 GPL-3.0 你可以：
- ✅ 自由使用、修改、再分发
- ✅ 商用（但必须开源衍生作品）
- ❌ 不可闭源
- ❌ 不可删除原作者署名

---

## 🐛 反馈

- **上游核心功能 bug** → 提到 [VisoMaster Issues](https://github.com/visomaster/VisoMaster/issues)
- **整合包脚本 / 启动器 / 中文化 bug** → 提到 [本仓库 Issues](https://github.com/wwzhifeng/VisoMaster-ZH/issues)
