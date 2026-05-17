# 开发者构建指南

从源码 build 完整整合包的详细步骤。

---

## 前置依赖

| 工具 | 用途 | 安装 |
|------|------|------|
| Git | clone 上游 | `winget install Git.Git` |
| 7-Zip | 打包成 7z | `winget install 7zip.7zip` |
| PowerShell | 跑 `.ps1` 脚本 | Windows 自带 |
| 网络 + 代理 | 拉 PyPI + GitHub | 必须 |
| 磁盘空间 | 至少 30GB | 用于 pip 包 + 模型 |

不需要预装 Python / CUDA / cuDNN —— 这些会被嵌入到包里。

---

## 一键构建（最常用）

```powershell
git clone https://github.com/wwzhifeng/VisoMaster-ZH.git
cd VisoMaster-ZH
powershell -ExecutionPolicy Bypass -File .\_internal\_build_package.ps1 -Stage all
```

完整流程会跑 5 个阶段，耗时约 60 分钟（视网速）：

| 阶段 | 做的事 | 时长 | 产物 |
|------|--------|------|------|
| python | 下载 Python 3.10.11 embed + bootstrap pip | 2 分钟 | `python/` |
| deps | pip 装 torch/TRT/ORT 等 8GB 依赖 | 20~40 分钟 | `python/Lib/site-packages/` |
| source | git clone 上游 VisoMaster 到 `app/` | 1 分钟 | `app/` + `main.py` |
| assets | 校验 BAT/配置/启动器文件齐全 | 秒级 | （只验证） |
| pack | 7z 打包成发布产物 | 5~10 分钟 | `dist/*.7z` |

---

## 分阶段构建（调试时用）

```powershell
# 只重装依赖（torch 升级时）
.\_internal\_build_package.ps1 -Stage deps

# 只刷新上游源码
.\_internal\_build_package.ps1 -Stage source

# 只重新打包
.\_internal\_build_package.ps1 -Stage pack
```

---

## 配置项

`_internal/_build_package.ps1` 顶部支持几个参数：

```powershell
.\_internal\_build_package.ps1 `
    -Stage all `
    -PyVersion 3.10.11 `
    -VisoMasterRef main `
    -Force
```

- `-PyVersion` ：Python 版本（默认 3.10.11，3.10.12+ 没 embed 构建用不了）
- `-VisoMasterRef` ：上游 git ref（默认 main，可指定 tag/commit）
- `-Force` ：强制重新下载 Python embed

---

## 手动跳坑指南

### pip 装 cu128 慢

PyTorch cu128 源在国外，国内裸跑很慢。开代理：

```powershell
$env:HTTPS_PROXY = "http://127.0.0.1:10809"
$env:HTTP_PROXY  = "http://127.0.0.1:10809"
.\_internal\_build_package.ps1 -Stage deps
```

### TensorRT 装不上（wheel_stub 错误）

`tensorrt` 包是 stub，要从 NVIDIA 自家 pypi 拉真 wheel。脚本已经传 `--extra-index-url https://pypi.nvidia.com`，如果还是报错检查网络。

### Python 3.10.13 找不到 embed

3.10.11 之后官方只发源码，没 Windows embed 构建。**必须用 3.10.11**。

### git clone 上游慢

`-Stage source` 用 `--depth 1` 已经做了浅克隆。如果还是慢，加镜像：

```powershell
git config --global url."https://hub.gitmirror.com/https://github.com".insteadOf "https://github.com"
.\_internal\_build_package.ps1 -Stage source
```

完事记得把镜像 unset 回来。

### 7z 找不到

`-Stage pack` 需要 PATH 里有 `7z.exe`。装完 7-Zip 后重启 PowerShell。

---

## 生成模型清单（修改 manifest 时用）

如果上游 `models_data.py` 变了，需要重新生成 `_internal/manifest.json`：

```cmd
python\python.exe _internal\_gen_manifest.py
```

这会读 `app/processors/models_data.py` 抽取所有模型的 URL / SHA256，按 core / swapper / restorer / liveportrait / extra 分包写入 manifest。

---

## 发布到 GitHub Release（建议流程）

1. 本地 build 完，得到 `dist/VisoMaster-ZH-cu128-{日期}.7z`
2. 主包大小通常 5~7GB，**超过 GitHub 单文件限制（2GB）**
3. 用 7z 分卷压缩：
   ```cmd
   7z a -v1900m VisoMaster-ZH-cu128.7z VisoMaster-ZH-cu128-final\
   ```
4. 得到 `VisoMaster-ZH-cu128.7z.001` `.002` `.003`...
5. 全部传到 GitHub Release（每个 < 2GB 就能传）
6. 用户下载后用 7z 合并解压

模型包同理。
