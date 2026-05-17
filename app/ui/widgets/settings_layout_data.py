from app.ui.widgets.actions import control_actions
import cv2
from app.helpers.typing_helper import LayoutDictTypes
SETTINGS_LAYOUT_DATA: LayoutDictTypes = {
    'Appearance': {
        'ThemeSelection': {
            'level': 1,
            'label': '主题',
            'options': ['Dark', 'Dark-Blue', 'Light'],
            'default': 'Dark',
            'help': '选择要使用的主题',
            'exec_function': control_actions.change_theme,
            'exec_function_args': [],
        },
    },
    'General': {
        'ProvidersPrioritySelection': {
            'level': 1,
            'label': '推理引擎优先级',
            'options': ['CUDA', 'TensorRT', 'TensorRT-Engine', 'CPU'],
            'default': 'CUDA',
            'help': '选择系统使用的推理引擎优先级。',
            'exec_function': control_actions.change_execution_provider,
            'exec_function_args': [],
        },
        'nThreadsSlider': {
            'level': 1,
            'label': '线程数',
            'min_value': '1',
            'max_value': '30',
            'default': '2',
            'step': 1,
            'help': '设置播放和录制时的执行线程数，严重依赖 GPU 显存。',
            'exec_function': control_actions.change_threads_number,
            'exec_function_args': [],
        },
    },
    'Video Settings': {
        'VideoPlaybackCustomFpsToggle': {
            'level': 1,
            'label': '自定义视频播放帧率',
            'default': False,
            'help': '手动设置视频播放时使用的帧率',
            'exec_function': control_actions.set_video_playback_fps,
            'exec_function_args': [],
        },
        'VideoPlaybackCustomFpsSlider': {
            'level': 2,
            'label': '视频播放帧率',
            'min_value': '1',
            'max_value': '120',
            'default': '30',
            'parentToggle': 'VideoPlaybackCustomFpsToggle',
            'requiredToggleValue': True,
            'step': 1,
            'help': '设置播放视频时的最大帧率'
        },
    },
    'Auto Swap':{
        'AutoSwapToggle': {
            'level': 1,
            'label': '自动换脸',
            'default': False,
            'help': '加载视频/图片文件时自动使用选定的源人脸/嵌入进行换脸'
        },
    },
    'Detectors': {
        'DetectorModelSelection': {
            'level': 1,
            'label': '人脸检测模型',
            'options': ['RetinaFace', 'Yolov8', 'SCRFD', 'Yunet'],
            'default': 'RetinaFace',
            'help': '选择用于检测输入图片或视频中人脸的人脸检测模型。'
        },
        'DetectorScoreSlider': {
            'level': 1,
            'label': '检测置信度',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'help': '设置人脸检测的置信度阈值。数值越高检测越可靠，但可能漏掉一些人脸。'
        },
        'MaxFacesToDetectSlider': {
            'level': 1,
            'label': '最大检测人脸数',
            'min_value': '1',
            'max_value': '50',
            'default': '20',
            'step': 1,
            'help': '设置单帧中最多检测的人脸数量'

        },
        'AutoRotationToggle': {
            'level': 1,
            'label': '自动旋转',
            'default': False,
            'help': '自动旋转输入以检测不同方向的人脸。'
        },
        'ManualRotationEnableToggle': {
            'level': 1,
            'label': '手动旋转',
            'default': False,
            'help': '旋转人脸检测器以更好地检测不同角度的人脸。'
        },
        'ManualRotationAngleSlider': {
            'level': 2,
            'label': '旋转角度',
            'min_value': '0',
            'max_value': '270',
            'default': '0',
            'step': 90,
            'parentToggle': 'ManualRotationEnableToggle',
            'requiredToggleValue': True,
            'help': '设置输入人脸的角度，用于处理躺着/倒置等情况。角度按顺时针读取。'
        },
        'LandmarkDetectToggle': {
            'level': 1,
            'label': '启用人脸关键点检测',
            'default': False,
            'help': '启用或禁用人脸关键点检测，用于优化人脸对齐。'
        },
        'LandmarkDetectModelSelection': {
            'level': 2,
            'label': '关键点检测模型',
            'options': ['5', '68', '3d68', '98', '106', '203', '478'],
            'default': '203',
            'parentToggle': 'LandmarkDetectToggle',
            'requiredToggleValue': True,
            'help': '选择关键点检测模型，不同模型检测不同数量的人脸关键点。'
        },
        'LandmarkDetectScoreSlider': {
            'level': 2,
            'label': '关键点检测置信度',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentToggle': 'LandmarkDetectToggle',
            'requiredToggleValue': True,
            'help': '设置人脸关键点检测的置信度阈值。'
        },
        'DetectFromPointsToggle': {
            'level': 2,
            'label': '从关键点检测',
            'default': False,
            'parentToggle': 'LandmarkDetectToggle',
            'requiredToggleValue': True,
            'help': '启用以从指定的关键点检测人脸。'
        },
        'ShowLandmarksEnableToggle': {
            'level': 1,
            'label': '显示关键点',
            'default': False,
            'help': '实时显示人脸关键点。'
        },
        'ShowAllDetectedFacesBBoxToggle': {
            'level': 1,
            'label': '显示边界框',
            'default': False,
            'help': '为帧中所有检测到的人脸绘制边界框'
        }
    },
    'DFM Settings':{
        'MaxDFMModelsSlider':{
            'level': 1,
            'label': '最大 DFM 模型数',
            'min_value': '1',
            'max_value': '5',
            'default': '1',
            'step': 1,
            'help': "设置同时保存在内存中的最大 DFM 模型数量。请根据 GPU 显存设置。",
        }
    },
    'Frame Enhancer':{
        'FrameEnhancerEnableToggle':{
            'level': 1,
            'label': '启用帧增强器',
            'default': False,
            'help': '为视频输入启用帧增强以改善视觉质量。'
        },
        'FrameEnhancerTypeSelection':{
            'level': 2,
            'label': '帧增强器类型',
            'options': ['RealEsrgan-x2-Plus', 'RealEsrgan-x4-Plus', 'RealEsr-General-x4v3', 'BSRGan-x2', 'BSRGan-x4', 'UltraSharp-x4', 'UltraMix-x4', 'DDColor-Artistic', 'DDColor', 'DeOldify-Artistic', 'DeOldify-Stable', 'DeOldify-Video'],
            'default': 'RealEsrgan-x2-Plus',
            'parentToggle': 'FrameEnhancerEnableToggle',
            'requiredToggleValue': True,
            'help': '根据内容和分辨率需求选择要应用的帧增强类型。'
        },
        'FrameEnhancerBlendSlider': {
            'level': 2,
            'label': '混合',
            'min_value': '0',
            'max_value': '100',
            'default': '100',
            'step': 1,
            'parentToggle': 'FrameEnhancerEnableToggle',
            'requiredToggleValue': True,
            'help': '将增强结果混合回原始帧。'
        },
    },
    'Webcam Settings': {
        'WebcamMaxNoSelection': {
            'level': 2,
            'label': '最大摄像头数',
            'options': ['1', '2', '3', '4', '5', '6'],
            'default': '1',
            'help': '选择允许换脸的最大摄像头流数量。'
        },
        'WebcamBackendSelection': {
            'level': 2,
            'label': '摄像头后端',
            'options': ['Default', 'DirectShow', 'MSMF', 'V4L', 'V4L2', 'GSTREAMER'],
            'default': 'Default',
            'help': '选择用于访问摄像头的后端。'
        },
        'WebcamMaxResSelection': {
            'level': 2,
            'label': '摄像头分辨率',
            'options': ['480x360', '640x480', '1280x720', '1920x1080', '2560x1440', '3840x2160'],
            'default': '1280x720',
            'help': '选择摄像头的最大输入分辨率。'
        },
        'WebCamMaxFPSSelection': {
            'level': 2,
            'label': '摄像头帧率',
            'options': ['23', '30', '60'],
            'default': '30',
            'help': '设置摄像头的最大帧率 (FPS)。'
        },
    },
    'Virtual Camera': {
        'SendVirtCamFramesEnableToggle': {
            'level': 1,
            'label': '发送帧到虚拟摄像头',
            'default': False,
            'help': '将换脸后的视频/摄像头输出发送到虚拟摄像头，供外部应用程序使用',
            'exec_function': control_actions.toggle_virtualcam,
            'exec_function_args': [],
        },
        'VirtCamBackendSelection': {
            'level': 1,
            'label': '虚拟摄像头后端',
            'options': ['obs', 'unitycapture'],
            'default': 'obs',
            'help': '根据您设置的虚拟摄像头选择后端',
            'parentToggle': 'SendVirtCamFramesEnableToggle',
            'requiredToggleValue': True,
            'exec_function': control_actions.enable_virtualcam,
            'exec_funtion_args': [],
        },
    },
    'Face Recognition': {
        'RecognitionModelSelection': {
            'level': 1,
            'label': '识别模型',
            'options': ['Inswapper128ArcFace', 'SimSwapArcFace', 'GhostArcFace', 'CSCSArcFace'],
            'default': 'Inswapper128ArcFace',
            'help': '选择用于比较人脸相似度的 ArcFace 模型。'
        },
        'SimilarityTypeSelection': {
            'level': 1,
            'label': '换脸相似度类型',
            'options': ['Opal', 'Pearl', 'Optimal'],
            'default': 'Opal',
            'help': '选择换脸过程中人脸检测和匹配的相似度计算类型。'
        },
    },
    'Embedding Merge Method':{
        'EmbMergeMethodSelection':{
            'level': 1,
            'label': '嵌入合并方法',
            'options': ['Mean','Median'],
            'default': 'Mean',
            'help': '选择合并人脸嵌入的方法。"Mean" 取平均值，"Median" 取中位数，对异常值更具鲁棒性。'
        }
    },
    'Media Selection':{
        'TargetMediaFolderRecursiveToggle':{
            'level': 1,
            'label': '目标媒体包含子文件夹',
            'default': False,
            'help': '选择目标媒体文件夹时包含所有子文件夹中的文件'
        },
        'InputFacesFolderRecursiveToggle':{
            'level': 1,
            'label': '输入人脸包含子文件夹',
            'default': False,
            'help': '选择输入人脸文件夹时包含所有子文件夹中的文件'
        }
    }
}

CAMERA_BACKENDS = {
    'Default': cv2.CAP_ANY,
    'DirectShow': cv2.CAP_DSHOW,
    'MSMF': cv2.CAP_MSMF,
    'V4L': cv2.CAP_V4L,
    'V4L2': cv2.CAP_V4L2,
    'GSTREAMER': cv2.CAP_GSTREAMER,
}