from app.helpers import miscellaneous as misc_helpers
from app.ui.widgets.actions import layout_actions
from app.helpers.typing_helper import LayoutDictTypes

# Widgets in Face Swap tab are created from this Layout
SWAPPER_LAYOUT_DATA: LayoutDictTypes = {
    '换脸器': {
        'SwapModelSelection': {
            'level': 1,
            'label': '换脸模型',
            'options': ['Inswapper128', 'InStyleSwapper256 Version A', 'InStyleSwapper256 Version B', 'InStyleSwapper256 Version C', 'DeepFaceLive (DFM)', 'SimSwap512', 'GhostFace-v1', 'GhostFace-v2', 'GhostFace-v3', 'CSCS'],            'default': 'Inswapper128',
            'help': '选择用于换脸的换脸模型。'
        },
        'SwapperResSelection': {
            'level': 2,
            'label': '换脸分辨率',
            'options': ['128', '256', '384', '512'],
            'default': '128',
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'Inswapper128',
            'help': '选择换脸的分辨率（像素）。数值越高效果越好但处理速度越慢。'
        },
        'DFMModelSelection': {
            'level': 2,
            'label': 'DFM 模型',
            'options': misc_helpers.get_dfm_models_selection_values,
            'default': misc_helpers.get_dfm_models_default_value,
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'DeepFaceLive (DFM)',
            'help': '选择用于换脸的预训练 DeepFaceLive (DFM) 模型。'
        },
        'DFMAmpMorphSlider': {
            'level': 2,
            'label': 'AMP 变形因子',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'DeepFaceLive (DFM)',
            'help': 'DFM AMP 模型的 AMP 变形因子',
        },
        'DFMRCTColorToggle': {
            'level': 2,
            'label': 'RCT 颜色迁移',
            'default': False,
            'parentSelection': 'SwapModelSelection',
            'requiredSelectionValue': 'DeepFaceLive (DFM)',
            'help': 'DFM 模型的 RCT 颜色迁移',
        }
    },
    '人脸关键点校正': {
        'FaceAdjEnableToggle': {
            'level': 1,
            'label': '人脸调整',
            'default': False,
            'help': '这是一个实验性功能，用于对检测器找到的人脸关键点进行直接调整。还可以调整换脸的大小比例。'
        },
        'KpsXSlider': {
            'level': 2,
            'label': '关键点 X轴',
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
            'label': '关键点 Y轴',
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
            'help': '增大或缩小检测点之间的距离。'
        },
        'FaceScaleAmountSlider': {
            'level': 2,
            'label': '人脸缩放程度',
            'min_value': '-20',
            'max_value': '20',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceAdjEnableToggle',
            'requiredToggleValue': True,
            'help': '放大或缩小整张人脸。'
        },
        'LandmarksPositionAdjEnableToggle': {
            'level': 1,
            'label': '5点关键点调整',
            'default': False,
            'help': '这是一个实验性功能，用于对检测器找到的人脸关键点位置进行直接调整。'
        },
        'EyeLeftXAmountSlider': {
            'level': 2,
            'label': '左眼:   X',
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
            'label': '左眼:   Y',
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
            'label': '右眼:   X',
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
            'label': '右眼:   Y',
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
            'label': '鼻子:   X',
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
            'label': '鼻子:   Y',
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
            'label': '左嘴角:   X',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': '左右移动左嘴角检测点。'
        },
        'MouthLeftYAmountSlider': {
            'level': 2,
            'label': '左嘴角:   Y',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': '上下移动左嘴角检测点。'
        },
        'MouthRightXAmountSlider': {
            'level': 2,
            'label': '右嘴角:   X',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': '左右移动右嘴角检测点。'
        },
        'MouthRightYAmountSlider': {
            'level': 2,
            'label': '右嘴角:   Y',
            'min_value': '-100',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'LandmarksPositionAdjEnableToggle',
            'requiredToggleValue': True,
            'help': '上下移动右嘴角检测点。'
        },
    },
    '人脸相似度': {
        'SimilarityThresholdSlider': {
            'level': 1,
            'label': '相似度阈值',
            'min_value': '1',
            'max_value': '100',
            'default': '60',
            'step': 1,
            'help': '设置相似度阈值以控制检测到的人脸与参考（目标）人脸的相似程度。'
        },
        'StrengthEnableToggle': {
            'level': 1,
            'label': '强度',
            'default': False,
            'help': '应用额外的换脸迭代以增加结果强度，可能提高相似度。'
        },
        'StrengthAmountSlider': {
            'level': 2,
            'label': '程度',
            'min_value': '0',
            'max_value': '500',
            'default': '100',
            'step': 25,
            'parentToggle': 'StrengthEnableToggle',
            'requiredToggleValue': True,
            'help': '最多增加 5 倍额外换脸（500%）。200% 通常效果较好。设置为 0 可关闭换脸但允许管道其余部分应用于原始图像。'
        },
        'FaceLikenessEnableToggle': {
            'level': 1,
            'label': '人脸相似度',
            'default': False,
            'help': '这是一个用于直接调整人脸相似度的功能。'
        },
        'FaceLikenessFactorDecimalSlider': {
            'level': 2,
            'label': '程度',
            'min_value': '-1.00',
            'max_value': '1.00',
            'default': '0.00',
            'decimals': 2,
            'step': 0.05,
            'parentToggle': 'FaceLikenessEnableToggle',
            'requiredToggleValue': True,
            'help': '确定源人脸与目标人脸之间的相似度因子。'
        },
        'DifferencingEnableToggle': {
            'level': 1,
            'label': '差异处理',
            'default': False,
            'help': '当两张图片差异较小时，允许部分原始人脸显示在换脸结果中。有助于为换脸后人脸恢复一些纹理。'
        },
        'DifferencingAmountSlider': {
            'level': 2,
            'label': '程度',
            'min_value': '0',
            'max_value': '100',
            'default': '4',
            'step': 1,
            'parentToggle': 'DifferencingEnableToggle',
            'requiredToggleValue': True,
            'help': '数值越高，相似度约束越宽松。'
        },
        'DifferencingBlendAmountSlider': {
            'level': 2,
            'label': '混合程度',
            'min_value': '0',
            'max_value': '100',
            'default': '5',
            'step': 1,
            'parentToggle': 'DifferencingEnableToggle',
            'requiredToggleValue': True,
            'help': '差异混合值。'
        },
    },
    '人脸遮罩':{
        'BorderBottomSlider':{
            'level': 1,
            'label': '底部边界',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'help': '具有可调底部、左侧、右侧、顶部和侧边的矩形，将换脸结果遮罩回原始图像中。'
        },
        'BorderLeftSlider':{
            'level': 1,
            'label': '左侧边界',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'help': '具有可调底部、左侧、右侧、顶部和侧边的矩形，将换脸结果遮罩回原始图像中。'
        },
        'BorderRightSlider':{
            'level': 1,
            'label': '右侧边界',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'help': '具有可调底部、左侧、右侧、顶部和侧边的矩形，将换脸结果遮罩回原始图像中。'
        },
        'BorderTopSlider':{
            'level': 1,
            'label': '顶部边界',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'help': '具有可调底部、左侧、右侧、顶部和侧边的矩形，将换脸结果遮罩回原始图像中。'
        },
        'BorderBlurSlider':{
            'level': 1,
            'label': '边界模糊',
            'min_value': '0',
            'max_value': '100',
            'default': '10',
            'step': 1,
            'help': '边界遮罩混合距离。'
        },
        'OccluderEnableToggle': {
            'level': 1,
            'label': '遮挡遮罩',
            'default': False,
            'help': '允许遮挡人脸的对象显示在换脸图像中。'
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
            'help': '增大或缩小遮挡区域'
        },
        'DFLXSegEnableToggle': {
            'level': 1,
            'label': 'DFL XSeg 遮罩',
            'default': False,
            'help': '允许遮挡人脸的对象显示在换脸图像中。'
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
            'help': '增大或缩小遮挡区域。'
        },
        'OccluderXSegBlurSlider': {
            'level': 1,
            'label': '遮挡/DFL XSeg 模糊',
            'min_value': '0',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'parentToggle': 'OccluderEnableToggle | DFLXSegEnableToggle',
            'requiredToggleValue': True,
            'help': '遮挡和 XSeg 的混合值。'
        },
        'ClipEnableToggle': {
            'level': 1,
            'label': '文字遮罩',
            'default': False,
            'help': '使用文字描述来识别将出现在最终换脸图像中的对象。'
        },
        'ClipText': {
            'level': 2,
            'label': '文字遮罩输入',
            'min_value': '0',
            'max_value': '1000',
            'default': '',
            'width': 130,
            'parentToggle': 'ClipEnableToggle',
            'requiredToggleValue': True,
            'help': '使用时，在框中输入用逗号分隔的词语，然后按 <回车>。'
        },
        'ClipAmountSlider': {
            'level': 2,
            'label': '程度',
            'min_value': '0',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentToggle': 'ClipEnableToggle',
            'requiredToggleValue': True,
            'help': '增大以增强效果。'
        },
        'FaceParserEnableToggle': {
            'level': 1,
            'label': '人脸解析遮罩',
            'default': False,
            'help': '允许原始图像中未处理的背景显示在最终换脸结果中。'
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
            'help': '负值/正值缩小和放大遮罩。'
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
            'help': '调整遮罩大小。覆盖整张面部。'
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
            'help': '调整遮罩大小。覆盖左眉毛。'
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
            'help': '调整遮罩大小。覆盖右眉毛。'
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
            'help': '调整遮罩大小。覆盖左眼。'
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
            'help': '调整遮罩大小。覆盖右眼。'
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
            'help': '调整遮罩大小。覆盖眼镜。'
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
            'help': '调整遮罩大小。覆盖鼻子。'
        },
        'MouthParserSlider': {
            'level': 2,
            'label': '嘴巴',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': '调整遮罩大小。覆盖口腔内部，包括舌头。'
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
            'help': '调整遮罩大小。覆盖上唇。'
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
            'help': '调整遮罩大小。覆盖下唇。'
        },
        'NeckParserSlider': {
            'level': 2,
            'label': '脖子',
            'min_value': '0',
            'max_value': '30',
            'default': '0',
            'step': 1,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': '调整遮罩大小。覆盖脖子。'
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
            'help': '调整遮罩大小。覆盖头发。'
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
            'help': '背景解析器的混合值'
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
            'help': '面部解析器的混合值'
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
            'help': '红色调整'
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
            'help': '绿色调整'
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
            'help': '蓝色调整'
        },
        'FaceParserHairMakeupBlendAmountDecimalSlider': {
            'level': 3,
            'label': '混合程度',
            'min_value': '0.1',
            'max_value': '1.0',
            'default': '0.2',
            'step': 0.1,
            'decimals': 1,
            'parentToggle': 'FaceParserEnableToggle & FaceParserHairMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': '混合程度：0.0 表示原始颜色，1.0 表示完全目标颜色。'
        },
        'FaceParserLipsMakeupEnableToggle': {
            'level': 2,
            'label': '嘴唇化妆',
            'default': False,
            'parentToggle': 'FaceParserEnableToggle',
            'requiredToggleValue': True,
            'help': '启用嘴唇化妆'
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
            'help': '红色调整'
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
            'help': '绿色调整'
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
            'help': '蓝色调整'
        },
        'FaceParserLipsMakeupBlendAmountDecimalSlider': {
            'level': 3,
            'label': '混合程度',
            'min_value': '0.1',
            'max_value': '1.0',
            'default': '0.2',
            'step': 0.1,
            'decimals': 1,
            'parentToggle': 'FaceParserEnableToggle & FaceParserLipsMakeupEnableToggle',
            'requiredToggleValue': True,
            'help': '混合程度：0.0 表示原始颜色，1.0 表示完全目标颜色。'
        },
        'RestoreEyesEnableToggle': {
            'level': 1,
            'label': '还原眼睛',
            'default': False,
            'help': '从原始人脸还原眼睛。'
        },
        'RestoreEyesBlendAmountSlider': {
            'level': 2,
            'label': '眼睛混合程度',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': '增大以显示更多换脸后的眼睛。减小以显示更多原始眼睛。'
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
            'help': '当换脸远离画面时减小此值。'
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
            'help': '调整眼睛边界的混合。增大以显示更多原始眼睛。减小以显示更多换脸后的眼睛。'
        },
        'RestoreXEyesRadiusFactorDecimalSlider': {
            'level': 2,
            'label': 'X 眼睛半径因子',
            'min_value': '0.3',
            'max_value': '3.0',
            'default': '1.0',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': '此参数决定遮罩的形状。如果两个值都为 1.0，遮罩为圆形。如果任一值大于或小于 1.0，遮罩将沿相应方向拉伸或收缩变为椭圆形。'
        },
        'RestoreYEyesRadiusFactorDecimalSlider': {
            'level': 2,
            'label': 'Y 眼睛半径因子',
            'min_value': '0.3',
            'max_value': '3.0',
            'default': '1.0',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': '此参数决定遮罩的形状。如果两个值都为 1.0，遮罩为圆形。如果任一值大于或小于 1.0，遮罩将沿相应方向拉伸或收缩变为椭圆形。'
        },
        'RestoreXEyesOffsetSlider': {
            'level': 2,
            'label': 'X 眼睛偏移',
            'min_value': '-300',
            'max_value': '300',
            'default': '0',
            'step': 1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': '在 X 轴上移动眼睛遮罩。'
        },
        'RestoreYEyesOffsetSlider': {
            'level': 2,
            'label': 'Y 眼睛偏移',
            'min_value': '-300',
            'max_value': '300',
            'default': '0',
            'step': 1,
            'parentToggle': 'RestoreEyesEnableToggle',
            'requiredToggleValue': True,
            'help': '在 Y 轴上移动眼睛遮罩。'
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
            'help': '调整眼睛间距。'
        },
        'RestoreMouthEnableToggle': {
            'level': 1,
            'label': '还原嘴巴',
            'default': False,
            'help': '从原始人脸还原嘴巴。'
        },
        'RestoreMouthBlendAmountSlider': {
            'level': 2,
            'label': '嘴巴混合程度',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': '增大以显示更多换脸后的嘴巴。减小以显示更多原始嘴巴。'
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
            'help': '当换脸远离画面时增大此值。'
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
            'help': '调整嘴巴边界的混合。增大以显示更多原始嘴巴。减小以显示更多换脸后的嘴巴。'
        },
        'RestoreXMouthRadiusFactorDecimalSlider': {
            'level': 2,
            'label': 'X 嘴巴半径因子',
            'min_value': '0.3',
            'max_value': '3.0',
            'default': '1.0',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': '此参数决定遮罩的形状。如果两个值都为 1.0，遮罩为圆形。如果任一值大于或小于 1.0，遮罩将沿相应方向拉伸或收缩变为椭圆形。'
        },
        'RestoreYMouthRadiusFactorDecimalSlider': {
            'level': 2,
            'label': 'Y 嘴巴半径因子',
            'min_value': '0.3',
            'max_value': '3.0',
            'default': '1.0',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': '此参数决定遮罩的形状。如果两个值都为 1.0，遮罩为圆形。如果任一值大于或小于 1.0，遮罩将沿相应方向拉伸或收缩变为椭圆形。'
        },
        'RestoreXMouthOffsetSlider': {
            'level': 2,
            'label': 'X 嘴巴偏移',
            'min_value': '-300',
            'max_value': '300',
            'default': '0',
            'step': 1,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': '在 X 轴上移动嘴巴遮罩。'
        },
        'RestoreYMouthOffsetSlider': {
            'level': 2,
            'label': 'Y 嘴巴偏移',
            'min_value': '-300',
            'max_value': '300',
            'default': '0',
            'step': 1,
            'parentToggle': 'RestoreMouthEnableToggle',
            'requiredToggleValue': True,
            'help': '在 Y 轴上移动嘴巴遮罩。'
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
            'help': '调整遮罩边界的模糊。'
        },
    },

    '人脸颜色校正':{
        'AutoColorEnableToggle': {
            'level': 1,
            'label': '自动颜色迁移',
            'default': False,
            'help': '启用自动颜色迁移：1. Hans 测试无遮罩，2. Hans 测试有遮罩，3. DFL 方法无遮罩，4. DFL 原始方法。'
        },
        'AutoColorTransferTypeSelection':{
            'level': 2,
            'label': '迁移类型',
            'options': ['Test', 'Test_Mask', 'DFL_Test', 'DFL_Orig'],
            'default': 'Test',
            'parentToggle': 'AutoColorEnableToggle',
            'requiredToggleValue': True,
            'help': '选择自动颜色迁移方法类型。Hans 方法有时可能会产生一些伪影。'
        },
        'AutoColorBlendAmountSlider': {
            'level': 1,
            'label': '混合程度',
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
            'help': '微调换脸结果的 RGB 颜色值。'
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
            'help': '红色调整'
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
            'help': '绿色调整'
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
            'help': '蓝色调整'
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
            'label': '色相',
            'min_value': '-0.50',
            'max_value': '0.50',
            'default': '0.00',
            'step': 0.01,
            'decimals': 2,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': '调整色相。'
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
            'label': '噪点',
            'min_value': '0.0',
            'max_value': '20.0',
            'default': '0.0',
            'step': 0.5,
            'decimals': 1,
            'parentToggle': 'ColorEnableToggle',
            'requiredToggleValue': True,
            'help': '向换脸添加噪点。'
        },

        'JPEGCompressionEnableToggle': {
            'level': 1,
            'label': 'JPEG 压缩',
            'default': False,
            'help': '对换脸应用 JPEG 压缩以使输出更逼真',
        },
        'JPEGCompressionAmountSlider': {
            'level': 2,
            'label': '压缩程度',
            'min_value': '1',
            'max_value': '100',
            'default': '50',
            'step': 1,
            'parentToggle': 'JPEGCompressionEnableToggle',
            'requiredToggleValue': True,
            'help': '调整 JPEG 压缩程度'
        }
    },
    '混合调整':{
        'FinalBlendAdjEnableToggle': {
            'level': 1,
            'label': '最终混合',
            'default': False,
            'help': '在管道末端进行混合。'
        },
        'FinalBlendAmountSlider': {
            'level': 2,
            'label': '最终混合程度',
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
            'label': '整体遮罩混合程度',
            'min_value': '0',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'help': '组合遮罩混合距离。不应用于边界遮罩。'
        },
    },
}