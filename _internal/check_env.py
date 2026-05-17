"""
VisoMaster TRT Portable - 环境诊断
================================
输出当前整合包的完整运行环境快照, 便于排障。
"""
from __future__ import annotations

import json
import os
import sys
import shutil
import subprocess
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY_ROOT = ROOT / "python"
SITE = PY_ROOT / "Lib" / "site-packages"

# 复用 bootstrap 的 DLL 注册逻辑
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa
bootstrap._register_dll_dirs()


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def _ok(s):    return f"{GREEN}{s}{RESET}"
def _warn(s):  return f"{YELLOW}{s}{RESET}"
def _err(s):   return f"{RED}{s}{RESET}"
def _hdr(s):   return f"{CYAN}{s}{RESET}"


def _line(k, v, status="ok"):
    color = {"ok": _ok, "warn": _warn, "err": _err}.get(status, _ok)
    print(f"  {k:<18}{color(v)}")


def section(title: str):
    print()
    print(_hdr(f"── {title} " + "─" * (52 - len(title))))


def check_gpu():
    section("GPU & 驱动")
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,driver_version,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        for line in out.strip().splitlines():
            name, drv, mem, cap = [x.strip() for x in line.split(",")]
            _line("GPU", name)
            _line("驱动版本", drv)
            _line("显存", f"{mem} MiB")
            _line("计算能力", f"sm_{cap.replace('.', '')}")
        return True
    except FileNotFoundError:
        _line("nvidia-smi", "未找到 - 未安装 NVIDIA 驱动?", "err")
        return False
    except Exception as e:
        _line("nvidia-smi", str(e), "err")
        return False


def check_python():
    section("Python 运行时")
    _line("Python 版本", sys.version.split()[0])
    _line("可执行文件", sys.executable)
    _line("项目根", str(ROOT))
    _line("site-packages", str(SITE) if SITE.is_dir() else "缺失!",
          "ok" if SITE.is_dir() else "err")


def check_torch():
    section("PyTorch")
    try:
        import torch
        _line("torch", torch.__version__)
        _line("CUDA available", str(torch.cuda.is_available()),
              "ok" if torch.cuda.is_available() else "err")
        if torch.cuda.is_available():
            _line("torch.cuda.version", torch.version.cuda or "?")
            _line("cuDNN", str(torch.backends.cudnn.version() or "?"))
            _line("GPU 数量", str(torch.cuda.device_count()))
            for i in range(torch.cuda.device_count()):
                _line(f"  [{i}]", torch.cuda.get_device_name(i))
        return torch.cuda.is_available()
    except Exception as e:
        _line("torch", f"导入失败: {e}", "err")
        return False


def check_onnxruntime():
    section("ONNX Runtime")
    try:
        import onnxruntime as ort
        _line("onnxruntime", ort.__version__)
        provs = ort.get_available_providers()
        for p in ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]:
            status = "ok" if p in provs else "warn"
            _line(p, "可用" if p in provs else "不可用", status)
        return "CUDAExecutionProvider" in provs
    except Exception as e:
        _line("onnxruntime", f"导入失败: {e}", "err")
        return False


def check_tensorrt():
    section("TensorRT")
    try:
        import tensorrt as trt
        _line("tensorrt", trt.__version__)
        # 验证可创建 Builder
        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)
        _line("Builder 实例化", "OK")
        _line("最大 batch", str(builder.max_batch_size) if hasattr(builder, "max_batch_size") else "N/A")
        _line("平台 FP16 支持", str(builder.platform_has_fast_fp16))
        _line("平台 INT8 支持", str(builder.platform_has_fast_int8))
        return True
    except Exception as e:
        _line("tensorrt", f"导入失败: {e}", "err")
        return False


def check_models():
    section("模型清单")
    manifest_path = ROOT / "_internal" / "manifest.json"
    if not manifest_path.exists():
        _line("manifest.json", "缺失!", "err")
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        _line("manifest.json", f"解析失败: {e}", "err")
        return False
    total = 0
    present = 0
    by_pack = {}
    for pack_name, items in manifest.get("packs", {}).items():
        n_ok = 0
        for item in items:
            total += 1
            if (ROOT / item["path"]).exists():
                present += 1
                n_ok += 1
        by_pack[pack_name] = (n_ok, len(items))
    for pack_name, (a, b) in by_pack.items():
        st = "ok" if a == b else ("warn" if a > 0 else "err")
        _line(pack_name, f"{a}/{b}", st)
    _line("合计", f"{present}/{total}", "ok" if present == total else "warn")
    return present > 0


def check_engines():
    section("TRT 引擎")
    eng_root = ROOT / "engines"
    if not eng_root.exists():
        _line("engines/", "目录不存在", "warn")
        return False
    subs = [p for p in eng_root.iterdir() if p.is_dir()]
    if not subs:
        _line("engines/", "未编译任何引擎", "warn")
        return False
    for sub in subs:
        engines = list(sub.glob("*.engine"))
        _line(sub.name, f"{len(engines)} 个引擎")
    return True


def check_disk():
    section("磁盘空间")
    total, used, free = shutil.disk_usage(ROOT)
    gb = lambda n: f"{n / (1024**3):.1f} GB"
    _line("根所在盘", str(ROOT.anchor))
    _line("剩余空间", gb(free),
          "ok" if free > 5 * 1024**3 else "warn")


def main():
    print()
    print(_hdr("═" * 60))
    print(_hdr("  VisoMaster TRT Portable - 环境诊断"))
    print(_hdr("═" * 60))

    results = {
        "gpu":     check_gpu(),
        "python":  True and (check_python() or True),
        "torch":   check_torch(),
        "ort":     check_onnxruntime(),
        "trt":     check_tensorrt(),
        "models":  check_models(),
        "engines": check_engines(),
    }
    check_disk()

    section("总结")
    fatal = not (results["gpu"] and results["torch"] and results["trt"])
    if fatal:
        print(_err("  ✗ 关键组件缺失, 程序可能无法启动"))
    elif not results["models"]:
        print(_warn("  ⚠ 模型未就位, 请运行 Download_Models.bat"))
    elif not results["engines"]:
        print(_warn("  ⚠ TRT 引擎未编译, 首次启动将自动编译"))
    else:
        print(_ok("  ✓ 环境正常, 可正常使用"))
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
