"""
VisoMaster TRT Portable - 性能 benchmark
========================================
对比三种 provider 模式下,
跑同一段视频 (或同一张图 N 次) 的耗时与吞吐。

用法:
  python\\python.exe _internal\\_benchmark.py --input D:\\test.mp4 --frames 100

输出:
  - 每种 provider 的平均 FPS / 单帧延迟 / 显存峰值
  - 文本和 JSON 两种格式, 方便贴进 BENCHMARK.md
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import _internal.bootstrap as bs
bs._register_dll_dirs()
os.chdir(ROOT)


def get_gpu_mem_mb() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return 0


def run_one(provider: str, input_path: str, frames: int) -> dict:
    """
    对单个 provider 模式跑一次 benchmark。
    返回 {fps, latency_ms, vram_peak_mb, total_s}
    """
    print(f"\n=== Benchmark: provider={provider}, frames={frames} ===")
    import cv2
    import torch

    # 加载主程序需要的几个核心模块
    from app.processors.models_processor import ModelsProcessor

    # 自动 no-op 的 fake widget: 任何属性访问都返回可调用的空对象
    class _NoOp:
        def __init__(self, *a, **kw): pass
        def __call__(self, *a, **kw): return self
        def __getattr__(self, n): return _NoOp()
        def emit(self, *a, **kw): pass
        def connect(self, *a, **kw): pass

    class FakeWindow:
        def __init__(self):
            self.models_processor = None
            self.control = {
                'ProvidersPrioritySelection': provider,
                'DetectorModelSelection': 'RetinaFace',
                'DetectorScoreSlider': 50,
                'LandmarkDetectToggle': False,
                'LandmarkDetectModelSelection': '2DFAN4',
                'LandmarkDetectScoreSlider': 50,
                'DetectFromPointsToggle': False,
                'AutoRotationToggle': False,
                'MaxFacesToDetectSlider': 5,
            }
        def __getattr__(self, name):
            # 任何缺失属性 (signal / widget / 方法) 都返回 NoOp
            return _NoOp()
    fw = FakeWindow()
    mp = ModelsProcessor(fw)
    fw.models_processor = mp

    # 切 provider
    if hasattr(mp, "switch_providers_priority"):
        mp.switch_providers_priority(provider)

    # 读输入
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"  无法打开 {input_path}")
        return {}

    vram_peak = 0
    t_total = time.time()
    n_done = 0
    for i in range(frames):
        ok, frame = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            _ = mp.run_detect(
                frame_rgb, 'RetinaFace',
                max_num=5, score=0.5, input_size=(512, 512),
                use_landmark_detection=False,
                landmark_detect_mode='2DFAN4',
                landmark_score=0.5,
                from_points=False,
                rotation_angles=[0],
            )
        except Exception as e:
            print(f"  frame {i}: {type(e).__name__}: {e}")
            continue
        n_done += 1
        vram_peak = max(vram_peak, get_gpu_mem_mb())
    cap.release()
    elapsed = time.time() - t_total

    fps = n_done / elapsed if elapsed > 0 else 0
    latency = elapsed / n_done * 1000 if n_done > 0 else 0
    print(f"  完成 {n_done}/{frames} 帧, 耗时 {elapsed:.1f}s, "
          f"FPS {fps:.1f}, 单帧 {latency:.1f}ms, 显存峰值 {vram_peak}MB")
    return {
        "provider": provider,
        "frames_done": n_done,
        "total_s": round(elapsed, 2),
        "fps": round(fps, 2),
        "latency_ms": round(latency, 2),
        "vram_peak_mb": vram_peak,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="测试视频路径")
    ap.add_argument("--frames", type=int, default=100, help="跑多少帧")
    ap.add_argument("--providers", nargs="+",
                    default=["CUDA", "TensorRT", "TensorRT-Engine"],
                    help="测哪些 provider")
    ap.add_argument("--out", default="benchmark_result.json")
    args = ap.parse_args()

    results = []
    for p in args.providers:
        try:
            r = run_one(p, args.input, args.frames)
            if r:
                results.append(r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[{p}] 跳过: {e}")

    # 输出对比表
    print("\n" + "=" * 60)
    print(" Provider        FPS      Latency(ms)  VRAM(MB)  Total(s)")
    print("=" * 60)
    base_fps = results[0]["fps"] if results else 1
    for r in results:
        speedup = r["fps"] / base_fps if base_fps > 0 else 1
        print(f" {r['provider']:<14}  {r['fps']:>6.1f}   {r['latency_ms']:>8.1f}     "
              f"{r['vram_peak_mb']:>6d}    {r['total_s']:>6.1f}  ({speedup:.2f}x)")

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nJSON 输出: {args.out}")


if __name__ == "__main__":
    main()
