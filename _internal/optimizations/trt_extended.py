"""
优化 #3: 扩展 TRT 引擎到 Swapper / Restorer
==============================================
问题:
  原版只有 6 个 LivePortrait 模型走 TensorRTPredictor,
  Inswapper128 / GhostFace / SimSwap / GFPGAN / CodeFormer 等
  主力模型都跑在 ONNX Runtime CUDA EP 上, 浪费了 TRT 收益。

修复:
  - 把 models_data.models_trt_list 扩展到包含 swapper / restorer / enhancer
  - 修改 models_processor.load_model 的分发逻辑: 若有 .engine 文件存在,
    优先走 TensorRTPredictor; 否则 fallback 到 ORT CUDA EP

预期收益: 主流程 (检测 → 换脸 → 修复) 整体提速 30~60%

实现策略:
  1. 启动时根据 engines/ 目录下实际存在的 .engine 动态扩展
     models_trt_list 和 models_trt_path 映射
  2. 拦截 load_model_trt 让它能加载非 LivePortrait 模型
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def _detect_engine_dir() -> Path | None:
    """复用 bootstrap 的引擎子目录检测"""
    eng_root = ROOT / "engines"
    if not eng_root.exists():
        return None
    subs = sorted([p for p in eng_root.iterdir() if p.is_dir()],
                  key=lambda p: p.stat().st_mtime, reverse=True)
    return subs[0] if subs else None


# engine 文件名 -> 上游 models_data.py 的 model_name 映射
# 引擎文件名 = onnx stem + .engine (与 build_engines._gen_default_jobs 对齐)
ENGINE_TO_MODEL = {
    # === 换脸 swapper ===
    "inswapper_128.fp16.engine":               "Inswapper128",
    "InStyleSwapper256_Version_A.fp16.engine": "InStyleSwapper256 Version A",
    "InStyleSwapper256_Version_B.fp16.engine": "InStyleSwapper256 Version B",
    "InStyleSwapper256_Version_C.fp16.engine": "InStyleSwapper256 Version C",
    "simswap_512_unoff.engine":                "SimSwap512",
    "ghost_unet_1_block.engine":               "GhostFacev1",
    "ghost_unet_2_block.engine":               "GhostFacev2",
    "ghost_unet_3_block.engine":               "GhostFacev3",
    "cscs_256.engine":                         "CSCS",

    # === 人脸检测 detector ===
    "det_10g.engine":                          "RetinaFace",
    "scrfd_2.5g_bnkps.engine":                 "SCRFD2.5g",
    "yoloface_8n.engine":                      "YoloFace8n",
    "yunet_n_640_640.engine":                  "YunetN",

    # === 关键点 landmark ===
    "res50.engine":                            "FaceLandmark5",
    "2dfan4.engine":                           "FaceLandmark68",
    "1k3d68.engine":                           "FaceLandmark3d68",
    "peppapig_teacher_Nx3x256x256.engine":     "FaceLandmark98",
    "2d106det.engine":                         "FaceLandmark106",
    "landmark.engine":                         "FaceLandmark203",
    "face_landmarks_detector_Nx3x256x256.engine": "FaceLandmark478",
    "face_blendshapes_Nx146x2.engine":         "FaceBlendShapes",

    # === ArcFace 嵌入 ===
    "w600k_r50.engine":                        "Inswapper128ArcFace",
    "simswap_arcface_model.engine":            "SimSwapArcFace",
    "ghost_arcface_backbone.engine":           "GhostArcFace",
    "cscs_arcface_model.engine":               "CSCSArcFace",
    "cscs_id_adapter.engine":                  "CSCSIDArcFace",

    # === 人脸修复 restorer ===
    "GFPGANv1.4.engine":                       "GFPGANv1.4",
    "GPEN-BFR-256.engine":                     "GPENBFR256",
    "GPEN-BFR-512.engine":                     "GPENBFR512",
    "GPEN-BFR-1024.engine":                    "GPENBFR1024",
    "GPEN-BFR-2048.engine":                    "GPENBFR2048",
    "codeformer_fp16.engine":                  "CodeFormer",
    "VQFRv2.fp16.engine":                      "VQFRv2",
    "RestoreFormerPlusPlus.fp16.engine":       "RestoreFormerPlusPlus",

    # === 帧增强 enhancer ===
    "RealESRGAN_x2plus.fp16.engine":           "RealEsrganx2Plus",
    "RealESRGAN_x4plus.fp16.engine":           "RealEsrganx4Plus",
    "realesr-general-x4v3.engine":             "RealEsrx4v3",
    "BSRGANx2.fp16.engine":                    "BSRGANx2",
    "BSRGANx4.fp16.engine":                    "BSRGANx4",
    "4x-UltraSharp.fp16.engine":               "UltraSharpx4",
    "4x-UltraMix_Smooth.fp16.engine":          "UltraMixx4",

    # === 上色 colorizer ===
    "ColorizeArtistic.fp16.engine":            "DeoldifyArt",
    "ColorizeStable.fp16.engine":              "DeoldifyStable",
    "ColorizeVideo.fp16.engine":               "DeoldifyVideo",
    "ddcolor_artistic.engine":                 "DDColorArt",
    "ddcolor.engine":                          "DDcolor",

    # === 分割 mask ===
    "occluder.engine":                         "Occluder",
    "XSeg_model.engine":                       "XSeg",
    "faceparser_resnet34.engine":              "FaceParser",

    # === LivePortrait ===
    "motion_extractor.engine":                 "LivePortraitMotionExtractor",
    "appearance_feature_extractor.engine":     "LivePortraitAppearanceFeatureExtractor",
    "warping_spade-fix.engine":                "LivePortraitWarpingSpadeFix",
    "stitching.engine":                        "LivePortraitStitching",
    "stitching_eye.engine":                    "LivePortraitStitchingEye",
    "stitching_lip.engine":                    "LivePortraitStitchingLip",
}


def _build_runtime_engine_map() -> dict[str, str]:
    """扫描 engines/<GPU>/*.engine, 返回 {model_name: abs_path}"""
    eng_dir = _detect_engine_dir()
    if not eng_dir:
        return {}
    found = {}
    for engine_file in eng_dir.glob("*.engine"):
        model_name = ENGINE_TO_MODEL.get(engine_file.name)
        if model_name:
            found[model_name] = str(engine_file.resolve())
    return found


def _patch_models_data(engine_map: dict[str, str]) -> None:
    """
    向 models_data 注入扩展的 trt 模型清单。
    上游用 models_trt_list (list of dicts) + models_trt_path (model_name -> path)
    """
    md = importlib.import_module("app.processors.models_data")

    # 注入路径映射
    if hasattr(md, "models_trt_path") and isinstance(md.models_trt_path, dict):
        for name, path in engine_map.items():
            md.models_trt_path[name] = path
    else:
        # 上游若没这个变量, 创建一个供 models_processor 引用
        md.models_trt_path_extra = engine_map

    # 扩展 trt 列表 (供 UI / load 判断)
    if hasattr(md, "models_trt_list") and isinstance(md.models_trt_list, list):
        existing = set()
        for item in md.models_trt_list:
            if isinstance(item, dict) and "model_name" in item:
                existing.add(item["model_name"])
            elif isinstance(item, str):
                existing.add(item)
        for name, eng_path in engine_map.items():
            if name in existing:
                continue
            # 必须带 local_path 字段, 否则上游 ModelsProcessor.__init__ 会 KeyError
            md.models_trt_list.append({
                "model_name": name,
                "local_path": eng_path,
                "hash": "",
                "source": "visomaster-portable",
            })


def _patch_models_processor(engine_map: dict[str, str]) -> None:
    """
    拦截 models_processor.load_model:
      若 model_name 在 engine_map 中且对应 .engine 存在,
      直接走 TensorRTPredictor, 不再 fallback 到 ORT。
    """
    mp = importlib.import_module("app.processors.models_processor")

    # 找到 ModelsProcessor 类
    cls = None
    for name in dir(mp):
        obj = getattr(mp, name)
        if isinstance(obj, type) and "Processor" in name:
            cls = obj
            break
    if cls is None:
        raise RuntimeError("未找到 ModelsProcessor 类")

    orig_load = getattr(cls, "load_model", None)
    if orig_load is None:
        raise RuntimeError("ModelsProcessor 没有 load_model 方法")

    # TensorRTPredictor 应该在 mp 模块或其引用中
    TensorRTPredictor = getattr(mp, "TensorRTPredictor", None)
    if TensorRTPredictor is None:
        # 尝试从 utils 里找
        try:
            from app.processors.utils import tensorrt_predictor as tp_mod
            TensorRTPredictor = getattr(tp_mod, "TensorRTPredictor", None)
        except Exception:
            pass
    if TensorRTPredictor is None:
        raise RuntimeError("未找到 TensorRTPredictor")

    def patched_load(self, model_name, *args, **kwargs):
        engine_path = engine_map.get(model_name)
        if engine_path and os.path.exists(engine_path):
            # 已加载过则直接返回
            cache = getattr(self, "models", None)
            if cache is not None and model_name in cache and cache[model_name] is not None:
                return cache[model_name]
            predictor = TensorRTPredictor(
                model_path=engine_path,
                device=getattr(self, "device", "cuda"),
            )
            if cache is not None:
                cache[model_name] = predictor
            return predictor
        return orig_load(self, model_name, *args, **kwargs)

    patched_load.__wrapped_by__ = "trt_extended"
    cls.load_model = patched_load


def apply() -> None:
    engine_map = _build_runtime_engine_map()
    if not engine_map:
        print("    [trt_extended] 未发现可用 .engine, 跳过 (运行 Build_Engines.bat)")
        return
    _patch_models_data(engine_map)
    _patch_models_processor(engine_map)
    print(f"    [trt_extended] 注入 {len(engine_map)} 个 TRT 引擎: "
          f"{', '.join(list(engine_map)[:5])}{'...' if len(engine_map) > 5 else ''}")
