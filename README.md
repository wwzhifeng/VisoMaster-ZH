<div align="center">

# VisoMaster-ZH

**VisoMaster 中文整合包 · 一键启动 · 适配 RTX 50xx**

[![GitHub Release](https://img.shields.io/github/v/release/wwzhifeng/VisoMaster-ZH?style=flat-square)](https://github.com/wwzhifeng/VisoMaster-ZH/releases)
[![License](https://img.shields.io/badge/license-GPL--3.0-blue?style=flat-square)](LICENSE)
[![Stars](https://img.shields.io/github/stars/wwzhifeng/VisoMaster-ZH?style=flat-square)](https://github.com/wwzhifeng/VisoMaster-ZH/stargazers)
[![Forks](https://img.shields.io/github/forks/wwzhifeng/VisoMaster-ZH?style=flat-square)](https://github.com/wwzhifeng/VisoMaster-ZH/network/members)

[整合包下载](https://wangzhifeng.vip/) · [使用文档](#-快速开始) · [开发者构建](docs/BUILD.md) · [架构说明](docs/ARCHITECTURE.md)

</div>

---

## 📦 下载

> 完整整合包（含 Python 运行时 + 模型）请到下方下载：
>
> ### → [https://wangzhifeng.vip/](https://wangzhifeng.vip/)

代码仓库**不包含**主包（10~30GB）和模型权重，只含构建脚本与启动器源码。

---

## 🚀 快速开始

1. 从上方链接下载并解压主包（路径勿含中文 / 空格）
2. 双击 **`Start.bat`**
3. 在弹出的图形化启动器内：
   - 点 **下载模型** → 选 `core` → 等下载完成
   - 点 **编译 LP 引擎** → 等约 3 分钟
   - 点 **启动主程序**

完成。所有操作图形化，**无需命令行**。

---

## ✨ 核心特性

- **RTX 50xx 原生支持** — 升级 TensorRT 10.16+，原生 Blackwell sm_120
- **图形化启动器** — PySide6 GUI，替代命令行操作
- **完整中文化** — 主程序 UI 全部汉化
- **解压即用** — 嵌入式 Python + CUDA + cuDNN + TRT + ffmpeg，无需安装
- **智能模型管理** — 多镜像、断点续传、SHA256 校验
- **TRT 引擎自动编译** — 动态 shape / plugin / GPU 隔离全自动处理
- **完整诊断工具** — 出错可定位，闪退有日志

---

## 💻 系统要求

| 项 | 要求 |
|----|------|
| 操作系统 | Windows 10 / 11 (x64) |
| GPU | NVIDIA RTX 20xx ~ RTX 50xx |
| 显存 | ≥ 8GB（推荐 12GB+） |
| 驱动 | 555+（推荐 591+） |
| 磁盘 | ≥ 30GB 可用空间 |

---

## 🛠 开发者：从源码构建

```powershell
git clone https://github.com/wwzhifeng/VisoMaster-ZH.git
cd VisoMaster-ZH
powershell -ExecutionPolicy Bypass -File .\_internal\_build_package.ps1 -Stage all
```

详细步骤见 [docs/BUILD.md](docs/BUILD.md)。

---

## 📂 仓库结构

```
VisoMaster-ZH/
├── _internal/         构建/启动/优化脚本
│   ├── launcher/      启动器 GUI
│   ├── optimizations/ 运行时优化框架
│   └── _build_package.ps1
├── app/               主程序源码（含中文化）
├── docs/              文档
├── Start.bat          用户入口
└── README.md
```

---

## 🤝 致谢

本项目基于 [VisoMaster](https://github.com/visomaster/VisoMaster) (GPL-3.0)。

感谢上游作者及所有为换脸 / 表情驱动模型做出贡献的研究者。

---

## 📜 许可证

[GPL-3.0](LICENSE) — 与上游一致。允许使用、修改、再分发，禁止闭源。

---

## 🐛 反馈

- 上游功能 bug → [VisoMaster Issues](https://github.com/visomaster/VisoMaster/issues)
- 整合包 / 启动器 / 汉化 bug → [本仓库 Issues](https://github.com/wwzhifeng/VisoMaster-ZH/issues)

<div align="center">

**如果觉得有用，点个 ⭐ 支持一下**

</div>
