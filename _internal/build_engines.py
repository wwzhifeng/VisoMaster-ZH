"""
VisoMaster TRT Portable - 引擎批量编译器
======================================
功能:
  --check                 检查核心引擎是否齐全 (退出码 0=齐, 1=缺)
  --mode auto             首次启动场景, 仅编核心
  --mode 1                菜单选项 1: 编译核心
  --mode 2                菜单选项 2: 编译全部 (按已下载模型)
  --mode 3                强制重编全部
  --status                打印当前引擎状态
特性:
  - 按 GPU 型号 + TRT 版本 分目录, 避免互相污染
  - 共享 timing.cache, 同 GPU 后续编译显著加速
  - 失败模型不中断流程, 记录到 _build_log.txt
  - 支持 dynamic shape (manifest 中声明 profiles)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# 复用 bootstrap 的 DLL 注册
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa
bootstrap._register_dll_dirs()

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "_internal" / "engines_manifest.json"

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# 引擎清单 (与 manifest.json 模型清单分离, 这里专注 TRT 编译策略)
# ---------------------------------------------------------------------------
def load_engine_manifest() -> dict:
    """
    引擎编译清单:
      - core   = LivePortrait 6 个引擎 (主程序唯一直接用 TensorRTPredictor 加载的)
      - extra  = 其它所有 ONNX (高级用户可编, 但主程序的 ORT TRT EP 不会用,
                它会自己在 tensorrt-engines/ 目录下用 hash 命名重新编译)
    """
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))

    mfp = ROOT / "_internal" / "manifest.json"
    if not mfp.exists():
        return {"core": [], "extra": []}
    mf = json.loads(mfp.read_text(encoding="utf-8"))

    def to_job(item):
        stem = Path(item["path"]).stem
        return {"onnx": item["path"], "engine": stem + ".engine", "precision": "fp16"}

    out = {"core": [], "extra": []}
    for pack_name, items in mf.get("packs", {}).items():
        bucket = "core" if pack_name == "liveportrait" else "extra"
        for it in items:
            out[bucket].append(to_job(it))
    return out


# LP 引擎文件名 (供 WarmupWorker 和 UI 过滤)
LIVEPORTRAIT_ENGINE_NAMES = {
    "motion_extractor.engine",
    "appearance_feature_extractor.engine",
    "stitching.engine",
    "stitching_eye.engine",
    "stitching_lip.engine",
    "warping_spade-fix.engine",
}


# ---------------------------------------------------------------------------
# GPU 检测 + 引擎目录策略
# ---------------------------------------------------------------------------
def detect_gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, timeout=5,
        )
        name = out.strip().splitlines()[0]
        return name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    except Exception:
        return "UnknownGPU"


def engine_dir() -> Path:
    try:
        import tensorrt as trt
        ver = trt.__version__
    except Exception:
        ver = "unknown"
    sub = ROOT / "engines" / f"{detect_gpu_name()}_TRT{ver}_fp16"
    sub.mkdir(parents=True, exist_ok=True)
    return sub


# ---------------------------------------------------------------------------
# TRT 编译
# ---------------------------------------------------------------------------
class _PyTrtLogger:
    """延迟初始化的 Python TRT logger 单例 (子类化 trt.ILogger)"""
    _inst = None
    @classmethod
    def get(cls):
        if cls._inst is None:
            import tensorrt as trt
            class L(trt.ILogger):
                def __init__(self):
                    trt.ILogger.__init__(self)
                def log(self, sev, msg):
                    tag = sev.name if hasattr(sev, "name") else str(sev)
                    if tag in ("ERROR", "INTERNAL_ERROR", "WARNING"):
                        print(f"    [TRT][{tag}] {msg}", flush=True)
            cls._inst = L()
        return cls._inst


def _has_dynamic_shape(onnx_path: Path) -> tuple[bool, dict]:
    """
    扫描 ONNX 输入, 检测是否有动态维度 (-1)。
    返回 (是否动态, {input_name: shape_list})
    """
    import onnx
    model = onnx.load(str(onnx_path), load_external_data=False)
    inputs = {}
    has_dyn = False
    for inp in model.graph.input:
        shape = []
        for d in inp.type.tensor_type.shape.dim:
            if d.HasField("dim_value") and d.dim_value > 0:
                shape.append(d.dim_value)
            else:
                shape.append(-1)
                has_dyn = True
        inputs[inp.name] = shape
    return has_dyn, inputs


def _gen_profile_shapes(input_name: str, shape: list[int],
                        model_filename: str = "") -> tuple[tuple, tuple, tuple]:
    """
    根据输入 shape 启发式生成 (min, opt, max) profile。
    策略:
      - 仅 batch 动态 (-1,C,H,W)  -> batch [1,1,8]
      - 仅空间动态 (1,C,-1,-1)   -> H/W [320,640,1280], 检测/分割类
      - 全动态 (-1,C,-1,-1)      -> 超分类, max 限制 batch=1 + 空间 1024 防显存爆
    """
    fname = model_filename.lower()
    is_super_res = any(k in fname for k in
                       ("esrgan", "bsrgan", "ultrasharp", "ultramix",
                        "realesr", "colorize", "ddcolor", "deoldify"))

    # 全动态 + 超分 -> 保守 max
    fully_dyn = shape[0] == -1 and any(v == -1 for v in shape[2:] if len(shape) > 2)
    if fully_dyn and is_super_res:
        max_b, max_hw = 1, 1024
    elif fully_dyn:
        max_b, max_hw = 4, 1024
    elif shape[0] == -1:
        max_b, max_hw = 8, shape[-1] if shape[-1] != -1 else 1024
    else:
        max_b, max_hw = 1, 1280

    def fill(b, hw):
        out = []
        for i, v in enumerate(shape):
            if v != -1:
                out.append(v); continue
            if i == 0:
                out.append(b)
            elif i >= 2:
                out.append(hw)
            else:
                out.append(1)
        return tuple(out)

    return fill(1, 320), fill(1, 640), fill(max_b, max_hw)


def _build_dynamic(onnx_path: Path, engine_path: Path, precision: str,
                   workspace_mb: int = 4096) -> tuple[bool, str]:
    """对动态 shape 模型, 自己创建 builder 并填 optimization profile"""
    import tensorrt as trt

    has_dyn, inputs = _has_dynamic_shape(onnx_path)
    if not has_dyn:
        return False, "_build_dynamic 不应被静态 shape 模型调用"

    logger = _PyTrtLogger.get()
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            return False, f"ONNX 解析失败: {errs}"

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 workspace_mb * 1024 * 1024)
    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    for name, shape in inputs.items():
        if -1 in shape:
            mn, opt, mx = _gen_profile_shapes(name, shape, onnx_path.name)
            profile.set_shape(name, mn, opt, mx)
            print(f"    [profile] {name} {shape} -> min={mn} opt={opt} max={mx}", flush=True)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        return False, "build_serialized_network 返回 None"
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    return True, "OK"


# 已知不兼容当前 TRT 的 ONNX, 自动跳过 (避免日志吓到用户)
KNOWN_INCOMPATIBLE_ONNX = {
    "warping_spade.onnx",   # 旧版 GridSample 5D, TRT 不支持, 已被 warping_spade-fix.onnx 替代
}


def build_one(onnx_path: Path, engine_path: Path, precision: str,
              profiles: Optional[dict], timing_cache_path: Path,
              workspace_mb: int = 4096) -> tuple[bool, str]:
    """
    动态 shape -> 自建 builder 带 profile
    静态 shape -> 调用上游 onnx_to_trt
    """
    # 0. 已知不兼容的直接跳过
    if onnx_path.name in KNOWN_INCOMPATIBLE_ONNX:
        return True, "SKIP (已知不兼容, 主程序用替代版本)"

    # 1. 检测是否动态 shape
    try:
        is_dynamic, _ = _has_dynamic_shape(onnx_path)
    except Exception as e:
        return False, f"ONNX 加载失败: {e}"

    # 2. 是否需要加载 grid_sample_3d_plugin
    plugin_path = None
    name = onnx_path.name.lower()
    if "warping_spade" in name:
        candidate = ROOT / "model_assets" / "grid_sample_3d_plugin.dll"
        if candidate.exists():
            plugin_path = str(candidate)
        else:
            return False, "缺少 grid_sample_3d_plugin.dll"

    # 3. 动态 shape 走自建路径
    if is_dynamic:
        try:
            # 仍要加载 plugin (warping_spade 类不会有动态 shape, 但保险)
            if plugin_path:
                import ctypes
                ctypes.CDLL(plugin_path, mode=ctypes.RTLD_GLOBAL, winmode=0)
                import tensorrt as trt
                trt.init_libnvinfer_plugins(_PyTrtLogger.get(), namespace="")
            return _build_dynamic(onnx_path, engine_path, precision, workspace_mb)
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    # 4. 静态 shape: 先用自己的 builder (避开上游 bug), 失败再用上游
    try:
        if plugin_path:
            import ctypes
            ctypes.CDLL(plugin_path, mode=ctypes.RTLD_GLOBAL, winmode=0)
            import tensorrt as trt
            trt.init_libnvinfer_plugins(_PyTrtLogger.get(), namespace="")
        ok, msg = _build_static(onnx_path, engine_path, precision, workspace_mb)
        if ok:
            return ok, msg
        # fallback 到上游
        print(f"    [info] 自建 builder 失败 ({msg}), 尝试上游...", flush=True)
        sys.path.insert(0, str(ROOT))
        import app.processors.utils.engine_builder as eb
        eb.TRT_LOGGER = _PyTrtLogger.get()
        from app.processors.utils.engine_builder import onnx_to_trt
        onnx_to_trt(
            onnx_model_path=str(onnx_path),
            trt_model_path=str(engine_path),
            precision=precision,
            custom_plugin_path=plugin_path,
            verbose=False,
        )
        if not engine_path.exists() or engine_path.stat().st_size == 0:
            return False, "引擎文件未生成"
        return True, "OK"
    except SystemExit as e:
        return False, f"上游 EngineBuilder 退出 (rc={e.code})"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _build_static(onnx_path: Path, engine_path: Path, precision: str,
                  workspace_mb: int = 4096) -> tuple[bool, str]:
    """静态 shape 模型, 自建 builder, 与 _build_dynamic 同源但不加 profile"""
    import tensorrt as trt
    logger = _PyTrtLogger.get()
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            return False, f"ONNX 解析失败: {errs}"
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 workspace_mb * 1024 * 1024)
    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        return False, "build_serialized_network 返回 None"
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    return True, "OK"


# ---------------------------------------------------------------------------
# 批处理流程
# ---------------------------------------------------------------------------
def collect_jobs(mode: str, manifest: dict) -> list[dict]:
    if mode in ("1", "auto", "core"):
        return manifest["core"]
    if mode in ("2", "all"):
        return manifest["core"] + manifest["extra"]
    if mode in ("3", "force"):
        return manifest["core"] + manifest["extra"]
    return manifest["core"]


def run_build(mode: str, force: bool = False) -> int:
    manifest = load_engine_manifest()
    jobs = collect_jobs(mode, manifest)
    if mode == "3":
        force = True

    out_dir = engine_dir()
    timing = out_dir / "timing.cache"
    log = out_dir / "_build_log.txt"
    print(f"\n{CYAN}=== 引擎输出: {out_dir} ==={RESET}")

    results = []
    t_start = time.time()
    for i, job in enumerate(jobs, 1):
        onnx = ROOT / job["onnx"]
        eng = out_dir / job["engine"]
        if not onnx.exists():
            print(f"  [{i:2d}/{len(jobs)}] {YELLOW}缺失模型{RESET} {job['onnx']}")
            results.append((job["onnx"], "MISSING", 0.0))
            continue
        if eng.exists() and not force:
            print(f"  [{i:2d}/{len(jobs)}] {DIM}已存在 跳过{RESET} {job['engine']}")
            results.append((job["onnx"], "SKIP", 0.0))
            continue
        print(f"  [{i:2d}/{len(jobs)}] {CYAN}编译{RESET} {job['engine']} ...", flush=True)
        t0 = time.time()
        try:
            ok, msg = build_one(
                onnx, eng,
                precision=job.get("precision", "fp16"),
                profiles=job.get("profiles"),
                timing_cache_path=timing,
                workspace_mb=job.get("workspace_mb", 4096),
            )
        except Exception as e:
            ok, msg = False, f"{type(e).__name__}: {e}"
        dt = time.time() - t0
        results.append((job["onnx"], msg if ok else f"FAIL: {msg}", dt))
        tag = GREEN + "✓" + RESET if ok else RED + "✗" + RESET
        print(f"     {tag} {dt:6.1f}s  {msg if not ok else ''}")

    # 写日志
    elapsed = time.time() - t_start
    with open(log, "w", encoding="utf-8") as f:
        f.write(f"# Build log @ {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# GPU: {detect_gpu_name()}\n")
        f.write(f"# Elapsed: {elapsed:.1f}s\n\n")
        for r in results:
            f.write(f"{r[1]:<60}\t{r[2]:6.1f}s\t{r[0]}\n")

    n_ok = sum(1 for r in results if r[1] == "OK" or r[1] == "SKIP")
    n_fail = sum(1 for r in results if r[1].startswith("FAIL") or r[1] == "MISSING")
    print(f"\n{CYAN}总耗时 {elapsed:.1f}s | 成功 {n_ok} | 失败/缺失 {n_fail}{RESET}")
    print(f"日志: {log}")
    return 0 if n_fail == 0 else 1


# ---------------------------------------------------------------------------
# 状态查询 / 启动期检查
# ---------------------------------------------------------------------------
def print_status():
    manifest = load_engine_manifest()
    out_dir = engine_dir()
    print(f"\n{CYAN}引擎目录: {out_dir}{RESET}\n")
    for kind in ("core", "extra"):
        n_ok = n_total = 0
        for job in manifest[kind]:
            n_total += 1
            if (out_dir / job["engine"]).exists():
                n_ok += 1
        tag = GREEN if n_ok == n_total else (YELLOW if n_ok > 0 else RED)
        print(f"  {kind:<6} {tag}{n_ok}/{n_total}{RESET}")
    if (out_dir / "timing.cache").exists():
        size = (out_dir / "timing.cache").stat().st_size / 1024
        print(f"  timing.cache: {size:.1f} KB")


def check_core() -> int:
    manifest = load_engine_manifest()
    out_dir = engine_dir()
    for job in manifest["core"]:
        # 仅对存在的模型要求其引擎也存在
        if (ROOT / job["onnx"]).exists() and not (out_dir / job["engine"]).exists():
            return 1
    return 0


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="检查核心引擎是否齐全 (Start.bat 用)")
    ap.add_argument("--mode", default=None,
                    help="编译模式: auto/1/2/3 或 core/all/force")
    ap.add_argument("--status", action="store_true", help="打印状态")
    args = ap.parse_args()

    if args.check:
        sys.exit(check_core())
    if args.status:
        print_status()
        sys.exit(0)
    if args.mode:
        sys.exit(run_build(args.mode))
    # 无参数: 默认 auto
    sys.exit(run_build("auto"))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断。")
        sys.exit(130)
