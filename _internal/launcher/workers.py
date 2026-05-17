"""
启动器后台工作线程
==================
所有耗时操作 (下载/编译/诊断) 在 QThread 中跑, 通过信号回传日志和进度。
"""
from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Signal

ROOT = Path(__file__).resolve().parent.parent.parent


class LogStream:
    """把 print 输出重定向到 Qt signal"""
    def __init__(self, signal):
        self.signal = signal
        self._buf = ""

    def write(self, text: str):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = self._strip_ansi(line)
            if line.strip():
                self.signal.emit(line)

    def flush(self):
        if self._buf.strip():
            self.signal.emit(self._strip_ansi(self._buf))
            self._buf = ""

    @staticmethod
    def _strip_ansi(s: str) -> str:
        import re
        return re.sub(r"\x1b\[[0-9;]*m", "", s)


# ---------------------------------------------------------------------------
# 环境诊断 worker
# ---------------------------------------------------------------------------
class EnvCheckWorker(QThread):
    log = Signal(str)
    done = Signal(dict)  # {"gpu":..., "torch":..., "trt":..., "models":(ok,total), "engines":(ok,total)}

    def run(self):
        result = {
            "gpu_name": "未知", "gpu_mem": 0, "driver": "?",
            "torch_ver": "?", "trt_ver": "?", "ort_ver": "?",
            "models_ok": 0, "models_total": 0,
            "engines_ok": 0, "engines_total": 0,
        }
        try:
            # GPU
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                     "--format=csv,noheader,nounits"],
                    text=True, timeout=5,
                )
                parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
                result["gpu_name"] = parts[0]
                result["driver"] = parts[1]
                result["gpu_mem"] = int(parts[2])
                self.log.emit(f"✓ GPU: {parts[0]} / 驱动 {parts[1]} / {parts[2]} MB")
            except Exception as e:
                self.log.emit(f"✗ GPU 检测失败: {e}")

            # torch
            try:
                import torch
                result["torch_ver"] = torch.__version__
                self.log.emit(f"✓ PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
            except Exception as e:
                self.log.emit(f"✗ PyTorch 加载失败: {e}")

            # tensorrt
            try:
                import tensorrt as trt
                result["trt_ver"] = trt.__version__
                self.log.emit(f"✓ TensorRT: {trt.__version__}")
            except Exception as e:
                self.log.emit(f"✗ TensorRT 加载失败: {e}")

            # onnxruntime
            try:
                import onnxruntime as ort
                result["ort_ver"] = ort.__version__
                self.log.emit(f"✓ ONNX Runtime: {ort.__version__}")
            except Exception as e:
                self.log.emit(f"✗ ONNX Runtime 加载失败: {e}")

            # 模型统计
            import json
            mfp = ROOT / "_internal" / "manifest.json"
            if mfp.exists():
                mf = json.loads(mfp.read_text(encoding="utf-8"))
                total = ok = 0
                for items in mf["packs"].values():
                    for it in items:
                        total += 1
                        p = ROOT / it["path"]
                        if p.exists() and p.stat().st_size > 0:
                            ok += 1
                result["models_ok"] = ok
                result["models_total"] = total
                self.log.emit(f"✓ 模型: {ok}/{total}")

            # 引擎统计
            eng_root = ROOT / "engines"
            if eng_root.exists():
                engines = []
                for sub in eng_root.iterdir():
                    if sub.is_dir():
                        engines.extend(sub.glob("*.engine"))
                result["engines_ok"] = len(engines)
                # 计算"应有"数量比较麻烦, 先简化
                from _internal import build_engines as be
                em = be.load_engine_manifest()
                result["engines_total"] = len(em["core"]) + len(em["extra"])
                self.log.emit(f"✓ 引擎: {len(engines)}/{result['engines_total']}")
        except Exception:
            self.log.emit(traceback.format_exc())
        self.done.emit(result)


# ---------------------------------------------------------------------------
# 下载 worker
# ---------------------------------------------------------------------------
class DownloadWorker(QThread):
    log = Signal(str)
    progress = Signal(int, int)  # current, total (bytes 或文件数)
    done = Signal(bool)

    def __init__(self, pack_names: list[str], parent=None):
        super().__init__(parent)
        self.pack_names = pack_names

    def run(self):
        ok = True
        old_stdout = sys.stdout
        sys.stdout = LogStream(self.log)
        try:
            from _internal import model_manager as mm
            mf = mm.load_manifest()
            for pack in self.pack_names:
                self.log.emit(f"\n=== 开始下载: {pack} ===")
                if not mm.download_pack(pack, mf, max_workers=2):
                    ok = False
        except Exception:
            self.log.emit(traceback.format_exc())
            ok = False
        finally:
            sys.stdout.flush()
            sys.stdout = old_stdout
        self.done.emit(ok)


# ---------------------------------------------------------------------------
# 引擎编译 worker
# ---------------------------------------------------------------------------
class BuildEnginesWorker(QThread):
    log = Signal(str)
    done = Signal(bool)

    def __init__(self, mode: str, parent=None):
        super().__init__(parent)
        self.mode = mode  # core / all / force

    def run(self):
        ok = True
        old_stdout, old_stderr = sys.stdout, sys.stderr
        stream = LogStream(self.log)
        sys.stdout = stream
        sys.stderr = stream  # TRT logger 走 stderr, 必须同时重定向
        try:
            from _internal import build_engines as be
            rc = be.run_build(self.mode)
            ok = (rc == 0)
        except Exception:
            self.log.emit(traceback.format_exc())
            ok = False
        finally:
            try:
                sys.stdout.flush(); sys.stderr.flush()
            except Exception:
                pass
            sys.stdout, sys.stderr = old_stdout, old_stderr
        self.done.emit(ok)


# ---------------------------------------------------------------------------
# 启动主程序
# ---------------------------------------------------------------------------
class WarmupEnginesWorker(QThread):
    """
    预热所有 TRT 引擎: load 到 GPU 显存 + 创建 execution context。
    避免主程序首次切换 TensorRT-Engine 后第一次推理时的长时间冷启动。
    """
    log = Signal(str)
    done = Signal(bool)

    def run(self):
        ok = True
        old_stdout, old_stderr = sys.stdout, sys.stderr
        stream = LogStream(self.log)
        sys.stdout = stream
        sys.stderr = stream
        try:
            import time
            from pathlib import Path
            from _internal import build_engines as be
            eng_dir = be.engine_dir()
            # 只预热 LP 引擎 - 其它引擎主程序的 ORT TRT EP 不会用
            engines = sorted(
                e for e in eng_dir.glob("*.engine")
                if e.name in be.LIVEPORTRAIT_ENGINE_NAMES
            )
            if not engines:
                self.log.emit("没有可预热的 LivePortrait 引擎, 请先在启动器编译")
                ok = False
            else:
                import tensorrt as trt
                logger = trt.Logger(trt.Logger.ERROR)
                # 必须先加载 grid_sample plugin 才能 deserialize warping_spade-fix
                plugin = ROOT / "model_assets" / "grid_sample_3d_plugin.dll"
                if plugin.exists():
                    try:
                        import ctypes
                        ctypes.CDLL(str(plugin), mode=ctypes.RTLD_GLOBAL, winmode=0)
                        trt.init_libnvinfer_plugins(logger, namespace="")
                    except Exception as e:
                        self.log.emit(f"[warn] 加载 plugin 失败: {e}")
                runtime = trt.Runtime(logger)

                self.log.emit(f"预热 {len(engines)} 个引擎 (load + 创建 context)...")
                ok_count = 0
                t_total = time.time()
                for i, eng in enumerate(engines, 1):
                    t0 = time.time()
                    try:
                        with open(eng, "rb") as f:
                            data = f.read()
                        engine = runtime.deserialize_cuda_engine(data)
                        if engine is None:
                            self.log.emit(f"  [{i:2d}/{len(engines)}] ✗ {eng.name} (deserialize 失败)")
                            continue
                        ctx = engine.create_execution_context()
                        if ctx is None:
                            self.log.emit(f"  [{i:2d}/{len(engines)}] ✗ {eng.name} (context 失败)")
                            continue
                        dt = time.time() - t0
                        size_mb = eng.stat().st_size / (1024 * 1024)
                        self.log.emit(f"  [{i:2d}/{len(engines)}] ✓ {eng.name}  ({size_mb:.0f}MB, {dt:.1f}s)")
                        ok_count += 1
                        del ctx, engine  # 释放显存, 真正用时主程序会重新 load
                    except Exception as e:
                        self.log.emit(f"  [{i:2d}/{len(engines)}] ✗ {eng.name}: {type(e).__name__}: {e}")
                elapsed = time.time() - t_total
                self.log.emit(f"\n预热完成: {ok_count}/{len(engines)} | 总耗时 {elapsed:.1f}s")
                # 写 marker
                marker = eng_dir / ".warmup_done"
                marker.write_text(
                    f"{ok_count}/{len(engines)} engines warmed @ "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')}",
                    encoding="utf-8",
                )
                ok = ok_count > 0
        except Exception:
            self.log.emit(traceback.format_exc())
            ok = False
        finally:
            try:
                sys.stdout.flush(); sys.stderr.flush()
            except Exception:
                pass
            sys.stdout, sys.stderr = old_stdout, old_stderr
        self.done.emit(ok)


class MainAppLauncher(QThread):
    """启动 VisoMaster 主程序 (作为独立进程, 这样启动器可以保留)"""
    log = Signal(str)
    done = Signal(int)

    def run(self):
        py = ROOT / "python" / "python.exe"
        bs = ROOT / "_internal" / "bootstrap.py"
        env = os.environ.copy()
        # 不污染父进程 env
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            proc = subprocess.Popen(
                [str(py), str(bs)],
                cwd=str(ROOT), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            for line in proc.stdout:
                self.log.emit(line.rstrip())
            rc = proc.wait()
            self.done.emit(rc)
        except Exception as e:
            self.log.emit(f"启动失败: {e}")
            self.done.emit(-1)
