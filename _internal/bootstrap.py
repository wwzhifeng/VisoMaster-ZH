"""
VisoMaster TRT Portable - 启动引导
================================
职责:
  1. 在 import 业务代码之前显式注册所有 NVIDIA / TRT / torch DLL 目录
  2. 切换 cwd 到项目根 (原版用 ./model_assets 相对路径)
  3. 把 models/ 软链接 / 透传成原版期望的 ./model_assets/ 结构
  4. 调用 main.py 启动 Qt 应用
说明:
  Windows 10+ 后 ctypes / WinAPI 不再使用 PATH 搜索 DLL，
  必须用 os.add_dll_directory(); 这是嵌入式 Python 包能否正常加载
  CUDA / cuDNN / nvinfer DLL 的关键。
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# ------------------------------------------------------------
# 路径定位
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PY_ROOT = ROOT / "python"
SITE = PY_ROOT / "Lib" / "site-packages"

# ------------------------------------------------------------
# DLL 目录注册 (Windows only)
# ------------------------------------------------------------
def _register_dll_dirs() -> None:
    if sys.platform != "win32":
        return
    if not hasattr(os, "add_dll_directory"):
        return  # Python < 3.8, 不应发生

    # 顺序敏感: torch 内置 cuda 在前, pip nvidia-* 在后, tensorrt 最后
    candidates = [
        PY_ROOT,
        ROOT / "app" / "ffmpeg" / "bin",  # 嵌入式 ffmpeg
        SITE / "torch" / "lib",
        SITE / "nvidia" / "cuda_runtime" / "bin",
        SITE / "nvidia" / "cudnn" / "bin",
        SITE / "nvidia" / "cublas" / "bin",
        SITE / "nvidia" / "cufft" / "bin",
        SITE / "nvidia" / "curand" / "bin",
        SITE / "nvidia" / "cusparse" / "bin",
        SITE / "nvidia" / "cusolver" / "bin",
        SITE / "nvidia" / "nvjitlink" / "bin",
        SITE / "nvidia" / "nvtx" / "bin",
        SITE / "tensorrt_libs",
    ]
    registered = []
    for d in candidates:
        if d.is_dir():
            try:
                os.add_dll_directory(str(d))
                registered.append(d)
            except OSError:
                pass
    return registered


# ------------------------------------------------------------
# 兼容层: VisoMaster 原版用 ./model_assets/xxx 路径
# 我们的整合包用 ./models/{category}/xxx 结构,
# 在启动时把 model_assets/ 做成 junction 指向 models/
# ------------------------------------------------------------
def _setup_model_assets_link() -> None:
    """
    原版期望 ./model_assets/{liveportrait_onnx,faceswap_models,...}/xxx.onnx
    整合包改为分类结构。这里建立兼容映射:
      model_assets/liveportrait_onnx     -> models/liveportrait
      model_assets/faceswap_models       -> models/swapper
      model_assets/face_detectors        -> models/detector
      model_assets/landmark_models       -> models/landmark
      model_assets/face_restorers        -> models/restorer
      model_assets/frame_enhancers       -> models/enhancer
      model_assets/arcface_models        -> models/arcface
      model_assets/face_parsers          -> models/segment
      model_assets/colorizer_models      -> models/colorizer
    用 Windows directory junction 实现 (无需管理员权限)。
    """
    # 整合包直接用上游的 model_assets/ 名称, 不需要 junction
    # 保留函数是为了向后兼容, 实际什么都不做
    return


# ------------------------------------------------------------
# TRT 引擎目录映射: 把 engines/{GPU}_TRT{ver}_fp16 暴露成
# 原版期望的 ./tensorrt-engines (ORT TRT EP 也读这个目录)
# ------------------------------------------------------------
def _setup_engine_dir_link() -> None:
    import subprocess
    legacy = ROOT / "tensorrt-engines"
    if legacy.exists() or legacy.is_symlink():
        return
    # 先找到当前 GPU 对应的 engines 子目录
    try:
        import tensorrt as trt  # noqa
        sub = _detect_engine_subdir()
        if sub is None:
            return
        try:
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(legacy), str(sub)],
                check=True, capture_output=True, shell=False,
            )
        except Exception as e:
            print(f"[bootstrap] WARN: failed to junction tensorrt-engines: {e}")
    except Exception:
        # tensorrt 未安装或加载失败, 跳过 (主程序会报错)
        pass


def _detect_engine_subdir() -> Path | None:
    try:
        import tensorrt as trt
        # 用 ORT 拿 GPU 名比 pycuda 兼容性更好
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                # 通过 nvidia-smi 兜底
                pass
        except Exception:
            pass
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, timeout=5,
        )
        gpu_name = out.strip().splitlines()[0].replace(" ", "_").replace("/", "_")
        sub = ROOT / "engines" / f"{gpu_name}_TRT{trt.__version__}_fp16"
        return sub if sub.is_dir() else None
    except Exception:
        return None


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
def main() -> int:
    _register_dll_dirs()

    # 切到项目根, 业务代码用相对路径
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

    # 把嵌入式 ffmpeg/bin 加到 PATH (subprocess 调用 ffmpeg.exe 需要)
    ffmpeg_bin = ROOT / "app" / "ffmpeg" / "bin"
    if ffmpeg_bin.is_dir():
        cur_path = os.environ.get("PATH", "")
        if str(ffmpeg_bin) not in cur_path:
            os.environ["PATH"] = f"{ffmpeg_bin}{os.pathsep}{cur_path}"

    _setup_model_assets_link()
    _setup_engine_dir_link()

    # 在 main.py 加载业务代码之前应用优化 patch
    try:
        print("[bootstrap] 应用运行时优化:")
        from _internal.optimizations import apply_all
        apply_all(verbose=True)
    except Exception as e:
        print(f"[bootstrap] 优化框架加载失败 (不影响主程序): {e}")

    # 进入 main.py
    try:
        import runpy
        runpy.run_path(str(ROOT / "main.py"), run_name="__main__")
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 0
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
