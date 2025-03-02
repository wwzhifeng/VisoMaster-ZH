from app.helpers import miscellaneous as misc_helpers
from app.ui.widgets.actions import layout_actions
from app.helpers.typing_helper import LayoutDictTypes

# Widgets in Face Swap tab are created from this Layout
SWAPPER_LAYOUT_DATA: LayoutDictTypes = {
    'Swapper': {
        'SwapModelSelection': {
            'level': 1,
            'label': '交换模型',
            'options': ['Inswapper128', 'InStyleSwapper256 Version A', 'InStyleSwapper256 Version B', 'InStyleSwapper256 Version C', 'DeepFaceLive (DFM)', 'SimSwap512', 'GhostFace-v1', 'GhostFace-v2', 'GhostFace-v3', 'CSCS'],
            'default': 'Inswapper128',
            'help': '选择用于面部交换的交换模型。'
        },
        'SwapperResSelection': {
            'level': 2,
            'label': '交换分辨率',
            'options': ['128', '256', '384', '512'],
            'default': '128',
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'Inswapper128',
            'help': '选择交换面部的分辨率（像素）。较高的值提供更好的质量，但处理速度较慢。'
        },
        'DFMModelSelection': {
            'level': 2,
            'label': 'DFM模型',
            'options': misc_helpers.get_dfm_models_selection_values,
            'default': misc_helpers.get_dfm_models_default_value,
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'DeepFaceLive (DFM)',
            'help': '选择用于交换的预训练DeepFaceLive (DFM)模型。'
        },
        'DFMAmpMorphSlider': {
            'level': 2,
            'label': 'AMP变形因子',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'DeepFaceLive (DFM)',
            'help': 'DFM AMP模型的AMP变形因子'
        },
        'DFMRCTColorToggle': {
            'level': 2,
            'label': 'RCT颜色转移',
            'default': False,
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'DeepFaceLive (DFM)',
            'help': 'DFM模型的RCT颜色转移'
        }
    },
    'Face Landmarks Correction': {
        'FaceAdjEnableToggle': {
            'level': 1,
            'label': '面部调整',
            'default': False,
            'help': '这是一个实验性功能，用于直接调整检测器找到的面部关键点。还有一个选项可以调整交换面部的大小。'
        },
        'KpsXSlider': {
            'level': 2,
            'label': '关键点X轴',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceAdjEnableToggle',
            'requiredToggleValue': True,
            'help': '左右移动检测点。'
        },
        'KpsYSlider': {
            'level': 2,
            'label': '关键点Y轴',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceAdjEnableToggle',
            'requiredToggleValue': True,
            'help': '上下移动检测点。'
        },
        'KpsScaleSlider': {
            'level': 2,
            'label': '关键点缩放',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceAdjEnableToggle',
            'requiredToggleValue': True,
            'help': '放大或缩小检测点之间的距离。'
        },
        'FaceScaleAmountSlider': {
            'level': 2,
            'label': '面部缩放量',
            'min_value': '-20',
            'max_value': '20',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceAdjEnableToggle',
            'requiredToggleValue': True,
            'help': '放大或缩小整个面部。'
        },
        'LandmarksPositionAdjEnableToggle': {
            'level': 1,
            'label': '5点关键调整',
            'default': False,
            'help': '这是一个实验性功能，用于直接调整检测器找到的面部关键点的位置。'
        },
        'EyeLeftXAmountSlider': {
            'level': 2,
            'label': '左眼：X',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': '左右移动左眼检测点。'
        },
        'EyeLeftYAmountSlider': {
            'level': 2,
            'label': '左眼：Y',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': '上下移动左眼检测点。'
        },
        'EyeRightXAmountSlider': {
            'level': 2,
            'label': '右眼：X',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': '左右移动右眼检测点。'
        },
        'EyeRightYAmountSlider': {
            'level': 2,
            'label': '右眼：Y',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': '上下移动右眼检测点。'
        },
        'NoseXAmountSlider': {
            'level': 2,
            'label': '鼻子：X',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': '左右移动鼻子检测点。'
        },
        'NoseYAmountSlider': {
            'level': 2,
            'label': '鼻子：Y',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': '上下移动鼻子检测点。'
        },
        'MouthLeftXAmountSlider': {
            'level': 2,
            'label': '左嘴：X',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': '左右移动左嘴检测点。'
        },
        'MouthLeftYAmountSlider': {
            'level': 2,
            'label': '左嘴：Y',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': '上下移动左嘴检测点。'
        },
        'MouthRightXAmountSlider': {
            'level': 2,
            'label': '右嘴：X',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': '左右移动右嘴检测点。'
        },
        'MouthRightYAmountSlider': {
            'level': 2,
            'label': '右嘴：Y',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': '上下移动右嘴检测点。'
        },
    },
    'Face Similarity': {
        'SimilarityThresholdSlider': {
            'level': 1,
            'label': '相似度阈值',
            'min_value': '1',
            'max_value': '100',
            'default': '60',
            'step': 1,
            'help': '设置相似度阈值，控制检测到的面部与参考（目标）面部的相似程度。'
        },
        'StrengthEnableToggle': {
            'level': 1,
            'label': '强度',
            'default': False,
            'help': '应用额外的交换迭代以增强结果强度，可能会提高相似度。'
        },
        'StrengthAmountSlider': {
            'level': 2,
            'label': '强度量',
            'min_value': '0',
            'max_value': '500',
            'default': '100',
            'step': 25,
            'parentToggle': 'StrengthEnableToggle',
            'requiredToggleValue': True,
            'help': '最多增加5倍额外交换（500%）。200%通常是不错的结果。设置为0以关闭交换，但允许管道的其余部分应用于原始图像。'
        },
        'FaceLikenessEnableToggle': {
            'level': 1,
            'label': '面部相似度',
            'default': False,
            'help': '这是一个用于直接调整面部相似度的功能。'
        },
        'FaceLikenessFactorDecimalSlider': {
            'level': 2,
            'label': '相似度因子',
            'min_value': '-1.00',
            'max_value': '1.00',
            'default': '0.00',
            'decimals': 2,
            'step': 0.05,
            'parentToggle': 'FaceLikenessEnableToggle',
            'requiredToggleValue': True,
            'help': '确定源面部与指定面部之间的相似度因子。'
        },
        'DifferencingEnableToggle': {
            'level': 1,
            'label': '差异化',
            'default': False,
            'help': '当两张图像之间的差异较小时，允许原始面部在交换结果中显示部分内容。可帮助恢复交换面部的纹理。'
        },
        'DifferencingAmountSlider': {
            'level': 2,
            'label': '差异量',
            'min_value': '0',
            'max_value': '100',
            'default': '4',
            'step': 1,
            'parentToggle': 'DifferencingEnableToggle',
            'requiredToggleValue': True,
            'help': '较高的值会放宽相似度约束。'
        },
        'DifferencingBlendAmountSlider': {
            'level': 2,
            'label': '混合量',
            'min_value': '0',
            'max_value': '100',
            'default': '5',
            'step': 1,
            'parentToggle': 'DifferencingEnableToggle',
            'requiredToggleValue': True,
            'help': '差异化的混合值。'
        },
    },
    'Face Mask': {
        'BorderBottomSlider': {
            'level': 1,
            'label': '底部边界',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'help': '一个可调整底部、左侧、右侧、顶部和侧面的矩形，将交换后的面部结果蒙版回原始图像。'
        },
        'BorderLeftSlider': {
            'level': 1,
            'label': '左侧边界',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'help': '一个可调整底部、左侧、右侧、顶部和侧面的矩形，将交换后的面部结果蒙版回原始图像。'
        },
        'BorderRightSlider': {
            'level': 1,
            'label': '右侧边界',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'help': '一个可调整底部、左侧、右侧、顶部和侧面的矩形，将交换后的面部结果蒙版回原始图像。'
        },
        'BorderTopSlider': {
            'level': 1,
            'label': '顶部边界',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'help': '一个可调整底部、左侧、右侧、顶部和侧面的矩形，将交换后的面部结果蒙版回原始图像。'
        },
        'BorderBlurSlider': {
            'level': 1,
            'label': '边界模糊',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'help': '边界蒙版的混合距离。'
        },
        'OccluderEnableToggle': {
            'level': 1,
            'label': '遮挡蒙版',
            'default': False,
            'help': '允许遮挡面部的对象在交换后的图像中显示。'
        },
        'OccluderSizeSlider': {
            'level': 2,
            'label': '大小',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'OccluderEnableToggle',
            'requiredToggleValue': True,
            'help': '放大或缩小遮挡区域'
        },
        'DFLXSegEnableToggle': {
            'level': 1,
            'label': 'DFL XSeg蒙版',
            'default': False,
            'help': '允许遮挡面部的对象在交换后的图像中显示。'
        },
        'DFLXSegSizeSlider': {
            'level': 2,
            'label': '大小',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'DFLXSegEnableToggle',
            'requiredToggleValue': True,
            'help': '放大或缩小遮挡区域。'
        },
        'OccluderXSegBlurSlider': {
            'level': 1,
            'label': '遮挡/XSeg模糊',
            'min_value': '0',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'OccluderEnableToggle | DFLXSegEnableToggle',
            'requiredToggleValue': True,
            'help': '遮挡和XSeg的混合值。'
        },
        'ClipEnableToggle': {
            'level': 1,
            'label': '文本蒙版',
            'default': False,
            'help': '使用描述来识别将在最终交换图像中显示的对象。'
        },
        'ClipText': {
            'level': 2,
            'label': '文本蒙版输入',
            'min_value': '0',
            'max_value': '1000',
            'default': '',
            'width': 130,
            'parentToggle': 'ClipEnableToggle',
            'requiredToggleValue': True,
            'help': '使用方法：在框中输入单词，用逗号分隔，然后按<回车>。'
        },
        'ClipAmountSlider': {
            'level': 2,
            'label': '强度',
            'min_value': '0',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentToggle': 'ClipEnableToggle',
            'requiredToggleValue': True,
            'help': '增加以增强效果。'
        },
        'FaceParserEnableToggle': {
            'level': 1,
            'label': '面部分析蒙版',
            'default': False,
            'help': '允许未处理的原始图像背景在最终交换中显示。'
        },
        'BackgroundParserSlider': {
            'level': 2,
            'label': '背景',
            'min_value': '-50',
            'max_value': '50',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': '负值/正值缩小和放大蒙版。'
        },
        'FaceParserSlider': {
            'level': 2,
            'label': '面部',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': '调整蒙版大小。覆盖整个面部。'
        },
        'LeftEyebrowParserSlider': {
            'level': 2,
            'label': '左眉毛',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': '调整蒙版大小。覆盖左眉毛。'
        },
        'RightEyebrowParserSlider': {
            'level': 2,
            'label': '右眉毛',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': '调整蒙版大小。覆盖右眉毛。'
        },
        'LeftEyeParserSlider': {
            'level': 2,
            'label': '左眼',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': '调整蒙版大小。覆盖左眼。'
        },
        'RightEyeParserSlider': {
            'level': 2,
            'label': '右眼',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': '调整蒙版大小。覆盖右眼。'
        },
        'EyeGlassesParserSlider': {
            'level': 2,
            'label': '眼镜',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': '调整蒙版大小。覆盖眼镜。'
        },
        'NoseParserSlider': {
            'level': 2,
            'label': '鼻子',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': '调整蒙版大小。覆盖鼻子。'
        },
        'MouthParserSlider': {
            'level': 2,
            'label': '口腔',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': '调整蒙版大小。覆盖口腔内部，包括舌头。'
        },
        'UpperLipParserSlider': {
            'level': 2,
            'label': '上唇',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': '调整蒙版大小。覆盖上唇。'
        },
        'LowerLipParserSlider': {
            'level': 2,
            'label': '下唇',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': '调整蒙版大小。覆盖下唇。'
        },
        'NeckParserSlider': {
            'level': 2,
            'label': '颈部',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': '调整蒙版大小。覆盖颈部。'
        },
        'HairParserSlider': {
            'level': 2,
            'label': '头发',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': '调整蒙版大小。覆盖头发。'
        },
        'BackgroundBlurParserSlider': {
            'level': 2,
            'label': '背景模糊',
            'min_value': '0',
            'max_value': '100',
            'default': '5',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': '背景分析的混合值'
        },
        'FaceBlurParserSlider': {
            'level': 2,
            'label': '面部模糊',
            'min_value': '0',
            'max_value': '100',
            'default': '5',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': '面部分析的混合值'
        },
        'FaceParserHairMakeupEnableToggle': {
            'level': 2,
            'label': '头发化妆',
            'default': False,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': '启用头发化妆'
        },
        'FaceParserHairMakeupRedSlider': {
            'level': 3,
            'label': '红色',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle & FaceParserHairMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': '调整红色值'
        },
        'FaceParserHairMakeupGreenSlider': {
            'level': 3,
            'label': '绿色',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 3,
            'parentToggle': 'FaceParserEnableToggle & FaceParserHairMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': '调整绿色值'
        },
        'FaceParserHairMakeupBlueSlider': {
            'level': 3,
            'label': '蓝色',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle & FaceParserHairMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': '调整蓝色值'
        },
        'FaceParserHairMakeupBlendAmountDecimalSlider': {
            'level': 3,
            'label': '混合量',
            'min_value': '0.1',
            'max_value': '1.0',
            'default': '0.2',
            'step': 0.1,
            'decimals': 1,
            'parentToggle': 'FaceParserEnableToggle & FaceParserHairMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': '混合值：0.0表示原始颜色，1.0表示完全目标颜色。'
        },
        'FaceParserLipsMakeupEnableToggle': {
            'level': 2,
            'label': '唇部化妆',
            'default': False,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': '启用唇部化妆'
        },
        'FaceParserLipsMakeupRedSlider': {
            'level': 3,
            'label': '红色',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle & FaceParserLipsMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': '调整红色值'
        },
        'FaceParserLipsMakeupGreenSlider': {
            'level': 3,
            'label': '绿色',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 3,
            'parentToggle': 'FaceParserEnableToggle & FaceParserLipsMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': '调整绿色值'
        },
        'FaceParserLipsMakeupBlueSlider': {
            'level': 3,
            'label': '蓝色',
            'min_value': '0',
            'max_value': '255',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle & FaceParserLipsMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': '调整蓝色值'
        },
        'FaceParserLipsMakeupBlendAmountDecimalSlider': {
            'level': 3,
            'label': '混合量',
            'min_value': '0.1',
            'max_value': '1.0',
            'default': '0.2',
            'step': 0.1,
            'decimals': 1,
            'parentToggle': 'FaceParserEnableToggle & FaceParserLipsMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': '混合值：0.0表示原始颜色，1.0表示完全目标颜色。'
        },
        'RestoreEyesEnableToggle': {
            'level': 1,
            'label': '恢复眼睛',
            'default': False,
            'help': '从原始面部恢复眼睛。'
        },
        'RestoreEyesBlendAmountSlider': {
            'level': 2,
            'label': '眼睛混合量',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': '增加此值以显示更多交换后的眼睛，减少以显示更多原始眼睛。'
        },
        'RestoreEyesSizeFactorDecimalSlider': {
            'level': 2,
            'label': '眼睛大小因子',
            'min_value': '2.0',
            'max_value': '4.0',
            'default': '3.0',
            'decimals': 1,
            'step': 0.5,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': '当交换的面部在画面中缩小时减少此值。'
        },
        'RestoreEyesFeatherBlendSlider': {
            'level': 2,
            'label': '眼睛羽化混合',
            'min_value': '1',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': '调整眼睛边界的混合。增加此值以显示更多原始眼睛，减少以显示更多交换后的眼睛。'
        },
        'RestoreXEyesRadiusFactorDecimalSlider': {
            'level': 2,
            'label': 'X眼睛半径因子',
            'min_value': '0.3',
            'max_value': '3.0',
            'default': '1.0',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': '这些参数决定蒙版的形状。如果两者都为1.0，蒙版将为圆形。如果其中一个大于或小于1.0，蒙版将变为椭圆形，沿相应方向拉伸或收缩。'
        },
        'RestoreYEyesRadiusFactorDecimalSlider': {
            'level': 2,
            'label': 'Y眼睛半径因子',
            'min_value': '0.3',
            'max_value': '3.0',
            'default': '1.0',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': '这些参数决定蒙版的形状。如果两者都为1.0，蒙版将为圆形。如果其中一个大于或小于1.0，蒙版将变为椭圆形，沿相应方向拉伸或收缩。'
        },
        'RestoreXEyesOffsetSlider': {
            'level': 2,
            'label': 'X眼睛偏移',
            'min_value': '-300',
            'max_value': '300',
            'default': '0',
            'step': 1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': '在X轴上移动眼睛蒙版。'
        },
        'RestoreYEyesOffsetSlider': {
            'level': 2,
            'label': 'Y眼睛偏移',
            'min_value': '-300',
            'max_value': '300',
            'default': '0',
            'step': 1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': '在Y轴上移动眼睛蒙版。'
        },
        'RestoreEyesSpacingOffsetSlider': {
            'level': 2,
            'label': '眼睛间距偏移',
            'min_value': '-200',
            'max_value': '200',
            'default': '0',
            'step': 1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': '改变眼睛间距距离。'
        },
        'RestoreMouthEnableToggle': {
            'level': 1,
            'label': '恢复嘴巴',
            'default': False,
            'help': '从原始面部恢复嘴巴。'
        },
        'RestoreMouthBlendAmountSlider': {
            'level': 2,
            'label': '嘴巴混合量',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': '增加此值以显示更多交换后的嘴巴，减少以显示更多原始嘴巴。'
        },       
        'RestoreMouthSizeFactorSlider': {
            'level': 2,
            'label': '嘴巴大小因子',
            'min_value': '5',
            'max_value': '60',
            'default': '25',
            'step': 5,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': '当交换的面部在画面中缩小时增加此值。'
        },
        'RestoreMouthFeatherBlendSlider': {
            'level': 2,
            'label': '嘴巴羽化混合',
            'min_value': '1',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': '调整嘴巴边界的混合。增加此值以显示更多原始嘴巴，减少以显示更多交换后的嘴巴。'
        },
        'RestoreXMouthRadiusFactorDecimalSlider': {
            'level': 2,
            'label': 'X嘴巴半径因子',
            'min_value': '0.3',
            'max_value': '3.0',
            'default': '1.0',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': '这些参数决定蒙版的形状。如果两者都为1.0，蒙版将为圆形。如果其中一个大于或小于1.0，蒙版将变为椭圆形，沿相应方向拉伸或收缩。'
        },
        'RestoreYMouthRadiusFactorDecimalSlider': {
            'level': 2,
            'label': 'Y嘴巴半径因子',
            'min_value': '0.3',
            'max_value': '3.0',
            'default': '1.0',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': '这些参数决定蒙版的形状。如果两者都为1.0，蒙版将为圆形。如果其中一个大于或小于1.0，蒙版将变为椭圆形，沿相应方向拉伸或收缩。'
        },
        'RestoreXMouthOffsetSlider': {
            'level': 2,
            'label': 'X嘴巴偏移',
            'min_value': '-300',
            'max_value': '300',
            'default': '0',
            'step': 1,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': '在X轴上移动嘴巴蒙版。'
        },
        'RestoreYMouthOffsetSlider': {
            'level': 2,
            'label': 'Y嘴巴偏移',
            'min_value': '-300',
            'max_value': '300',
            'default': '0',
            'step': 1,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': '在Y轴上移动嘴巴蒙版。'
        },
        'RestoreEyesMouthBlurSlider': {
            'level': 1,
            'label': '眼睛/嘴巴模糊',
            'min_value': '0',
            'max_value': '50',
            'default': '0',
            'step': 1,
            'parentToggle': 'RestoreEyesEnableToggle | RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': '调整蒙版边界的模糊度。'
        },
    },
    'Face Color Correction': {
        'AutoColorEnableToggle': {
            'level': 1,
            'label': '自动颜色转移',
            'default': False,
            'help': '启用自动颜色转移：1. 无蒙版Hans测试，2. 有蒙版Hans测试，3. 无蒙版DFL方法，4. DFL原始方法。'
        },
        'AutoColorTransferTypeSelection': {
            'level': 2,
            'label': '转移类型',
            'options': ['Test', 'Test_Mask', 'DFL_Test', 'DFL_Orig'],
            'default': 'Test',
            'parentToggle': 'AutoColorEnableToggle',
            'requiredToggleValue': True,
            'help': '选择自动颜色转移的方法类型。Hans方法有时可能会有一些瑕疵。'
        },
        'AutoColorBlendAmountSlider': {
            'level': 1,
            'label': '混合量',
            'min_value': '0',
            'max_value': '100',
            'default': '80',
            'step': 5,
            'parentToggle': 'AutoColorEnableToggle',
            'requiredToggleValue': True,
            'help': '调整混合值。'
        },
        'ColorEnableToggle': {
            'level': 1,
            'label': '颜色调整',
            'default': False,
            'help': '微调交换的RGB颜色值。'
        },
        'ColorRedSlider': {
            'level': 1,
            'label': '红色',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': '调整红色值'
        },
        'ColorGreenSlider': {
            'level': 1,
            'label': '绿色',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': '调整绿色值'
        },
        'ColorBlueSlider': {
            'level': 1,
            'label': '蓝色',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': '调整蓝色值'
        },
        'ColorBrightnessDecimalSlider': {
            'level': 1,
            'label': '亮度',
            'min_value': '0.00',
            'max_value': '2.00',
            'default': '1.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': '调整亮度。'
        },
        'ColorContrastDecimalSlider': {
            'level': 1,
            'label': '对比度',
            'min_value': '0.00',
            'max_value': '2.00',
            'default': '1.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': '调整对比度。'
        },
        'ColorSaturationDecimalSlider': {
            'level': 1,
            'label': '饱和度',
            'min_value': '0.00',
            'max_value': '2.00',
            'default': '1.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': '调整饱和度。'
        },
        'ColorSharpnessDecimalSlider': {
            'level': 1,
            'label': '锐度',
            'min_value': '0.0',
            'max_value': '2.0',
            'default': '1.0',
            'step': 0.1,
            'decimals': 1,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': '调整锐度。'
        },
        'ColorHueDecimalSlider': {
            'level': 1,
            'label': '色调',
            'min_value': '-0.50',
            'max_value': '0.50',
            'default': '0.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': '调整色调。'
        },
        'ColorGammaDecimalSlider': {
            'level': 1,
            'label': '伽马',
            'min_value': '0.00',
            'max_value': '2.00',
            'default': '1.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': '调整伽马值。'
        },
        'ColorNoiseDecimalSlider': {
            'level': 1,
            'label': '噪声',
            'min_value': '0.0',
            'max_value': '20.0',
            'default': '0.0',
            'step': 0.5,
            'decimals': 1,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': '为交换的面部添加噪声。'
        },
        'JPEGCompressionEnableToggle': {
            'level': 1,
            'label': 'JPEG压缩',
            'default': False,
            'help': '对交换的面部应用JPEG压缩，使输出更真实'
        },
        'JPEGCompressionAmountSlider': {
            'level': 2,
            'label': '压缩量',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentToggle': 'JPEGCompressionEnableToggle',
            'requiredToggleValue': True,
            'help': '调整JPEG压缩量'
        }
    },
    'Blend Adjustments': {
        'FinalBlendAdjEnableToggle': {
            'level': 1,
            'label': '最终混合',
            'default': False,
            'help': '在管道末端进行混合。'
        },
        'FinalBlendAmountSlider': {
            'level': 2,
            'label': '最终混合量',
            'min_value': '1',
            'max_value': '50',
            'default': '1',
            'step': 1,
            'parentToggle': 'FinalBlendAdjEnableToggle',
            'requiredToggleValue': True,
            'help': '调整最终混合值。'
        },
        'OverallMaskBlendAmountSlider': {
            'level': 1,
            'label': '总体蒙版混合量',
            'min_value': '0',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'help': '组合蒙版的混合距离。不适用于边界蒙版。'
        },        
    },
}
