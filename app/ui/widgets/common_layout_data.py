from app.helpers.typing_helper import LayoutDictTypes
import app.ui.widgets.actions.layout_actions as layout_actions

COMMON_LAYOUT_DATA: LayoutDictTypes = {
    # '人脸对比':{
    #     'ViewFaceMaskEnableToggle':{
    #         'level': 1,
    #         'label': '查看人脸遮罩',
    #         'default': False,
    #         'help': '显示人脸遮罩',
    #         'exec_function': layout_actions.fit_image_to_view_onchange,
    #         'exec_function_args': [],
    #     },
    #     'ViewFaceCompareEnableToggle':{
    #         'level': 1,
    #         'label': '查看人脸对比',
    #         'default': False,
    #         'help': '显示人脸对比',
    #         'exec_function': layout_actions.fit_image_to_view_onchange,
    #         'exec_function_args': [],
    #     },
    # },
    '人脸修复': {
        'FaceRestorerEnableToggle': {
            'level': 1,
            'label': '启用人脸修复',
            'default': False,
            'help': '启用人脸修复模型以改善换脸后的面部质量。'
        },
        'FaceRestorerTypeSelection': {
            'level': 2,
            'label': '修复器类型',
            'options': ['GFPGAN-v1.4', 'CodeFormer', 'GPEN-256', 'GPEN-512', 'GPEN-1024', 'GPEN-2048', 'RestoreFormer++', 'VQFR-v2'],
            'default': 'GFPGAN-v1.4',
            'parentToggle': 'FaceRestorerEnableToggle',
            'requiredToggleValue': True,
            'help': '选择人脸修复的模型类型。'
        },
        'FaceRestorerDetTypeSelection': {
            'level': 2,
            'label': '对齐方式',
            'options': ['Original', 'Blend', 'Reference'],
            'default': 'Original',
            'parentToggle': 'FaceRestorerEnableToggle',
            'requiredToggleValue': True,
            'help': '选择将人脸恢复到原始位置或混合位置的对齐方法。'
        },
        'FaceFidelityWeightDecimalSlider': {
            'level': 2,
            'label': '保真度权重',
            'min_value': '0.0',
            'max_value': '1.0',
            'default': '0.9',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'FaceRestorerEnableToggle',
            'requiredToggleValue': True,
            'help': '调整保真度权重以控制修复保留原始人脸细节的程度。'
        },
        'FaceRestorerBlendSlider': {
            'level': 2,
            'label': '混合',
            'min_value': '0',
            'max_value': '100',
            'default': '100',
            'step': 1,
            'parentToggle': 'FaceRestorerEnableToggle',
            'requiredToggleValue': True,
            'help': '控制修复后人脸与换脸后人脸之间的混合比例。'
        },
        'FaceRestorerEnable2Toggle': {
            'level': 1,
            'label': '启用人脸修复 2',
            'default': False,
            'help': '启用人脸修复模型以改善换脸后的面部质量。'
        },
        'FaceRestorerType2Selection': {
            'level': 2,
            'label': '修复器类型',
            'options': ['GFPGAN-v1.4', 'CodeFormer', 'GPEN-256', 'GPEN-512', 'GPEN-1024', 'GPEN-2048', 'RestoreFormer++', 'VQFR-v2'],
            'default': 'GFPGAN-v1.4',
            'parentToggle': 'FaceRestorerEnable2Toggle',
            'requiredToggleValue': True,
            'help': '选择人脸修复的模型类型。'
        },
        'FaceRestorerDetType2Selection': {
            'level': 2,
            'label': '对齐方式',
            'options': ['Original', 'Blend', 'Reference'],
            'default': 'Original',
            'parentToggle': 'FaceRestorerEnable2Toggle',
            'requiredToggleValue': True,
            'help': '选择将人脸恢复到原始位置或混合位置的对齐方法。'
        },
        'FaceFidelityWeight2DecimalSlider': {
            'level': 2,
            'label': '保真度权重',
            'min_value': '0.0',
            'max_value': '1.0',
            'default': '0.9',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'FaceRestorerEnable2Toggle',
            'requiredToggleValue': True,
            'help': '调整保真度权重以控制修复保留原始人脸细节的程度。'
        },
        'FaceRestorerBlend2Slider': {
            'level': 2,
            'label': '混合',
            'min_value': '0',
            'max_value': '100',
            'default': '100',
            'step': 1,
            'parentToggle': 'FaceRestorerEnable2Toggle',
            'requiredToggleValue': True,
            'help': '控制修复后人脸与换脸后人脸之间的混合比例。'
        },
        'FaceExpressionEnableToggle': {
            'level': 1,
            'label': '启用面部表情修复',
            'default': False,
            'help': '启用 LivePortrait 面部表情模型以在换脸后恢复面部表情。'
        },
        'FaceExpressionCropScaleDecimalSlider': {
            'level': 2,
            'label': '裁剪比例',
            'min_value': '1.80',
            'max_value': '3.00',
            'default': '2.30',
            'step': 0.05,
            'decimals': 2,
            'parentToggle': 'FaceExpressionEnableToggle',
            'requiredToggleValue': True,
            'help': '更改换脸裁剪比例。增大数值可更远距离捕捉人脸。'
        },
        'FaceExpressionVYRatioDecimalSlider': {
            'level': 2,
            'label': 'VY 比率',
            'min_value': '-0.125',
            'max_value': '-0.100',
            'default': '-0.125',
            'step': 0.001,
            'decimals': 3,
            'parentToggle': 'FaceExpressionEnableToggle',
            'requiredToggleValue': True,
            'help': '更改裁剪比例的 vy 比率。增大数值可更远距离捕捉人脸。'
        },
        'FaceExpressionFriendlyFactorDecimalSlider': {
            'level': 2,
            'label': '表情相似度',
            'min_value': '0.0',
            'max_value': '1.0',
            'default': '1.0',
            'decimals': 1,
            'step': 0.1,
            'parentToggle': 'FaceExpressionEnableToggle',
            'requiredToggleValue': True,
            'help': '控制驱动人脸与换脸后人脸之间的表情相似度。'
        },
        'FaceExpressionAnimationRegionSelection': {
            'level': 2,
            'label': '动画区域',
            'options': ['all', 'eyes', 'lips'],
            'default': 'all',
            'parentToggle': 'FaceExpressionEnableToggle',
            'requiredToggleValue': True,
            'help': '修复过程中涉及的面部区域。'
        },
        'FaceExpressionNormalizeLipsEnableToggle': {
            'level': 2,
            'label': '标准化嘴唇',
            'default': True,
            'parentToggle': 'FaceExpressionEnableToggle',
            'requiredToggleValue': True,
            'help': '在面部修复过程中标准化嘴唇。'
        },
        'FaceExpressionNormalizeLipsThresholdDecimalSlider': {
            'level': 3,
            'label': '标准化嘴唇阈值',
            'min_value': '0.00',
            'max_value': '1.00',
            'default': '0.03',
            'decimals': 2,
            'step': 0.01,
            'parentToggle': 'FaceExpressionNormalizeLipsEnableToggle & FaceExpressionEnableToggle',
            'requiredToggleValue': True,
            'help': '标准化嘴唇的阈值。'
        },
        'FaceExpressionRetargetingEyesEnableToggle': {
            'level': 2,
            'label': '重定向眼睛',
            'default': False,
            'parentToggle': 'FaceExpressionEnableToggle',
            'requiredToggleValue': True,
            'help': '在面部修复过程中调整或重定向眼睛的视线或运动。此选项将覆盖动画区域设置，即动画区域将被忽略。'
        },
        'FaceExpressionRetargetingEyesMultiplierDecimalSlider': {
            'level': 3,
            'label': '重定向眼睛倍数',
            'min_value': '0.00',
            'max_value': '2.00',
            'default': '1.00',
            'decimals': 2,
            'step': 0.01,
            'parentToggle': 'FaceExpressionRetargetingEyesEnableToggle & FaceExpressionEnableToggle',
            'requiredToggleValue': True,
            'help': '重定向眼睛的倍数值。'
        },
        'FaceExpressionRetargetingLipsEnableToggle': {
            'level': 2,
            'label': '重定向嘴唇',
            'default': False,
            'parentToggle': 'FaceExpressionEnableToggle',
            'requiredToggleValue': True,
            'help': '在面部修复过程中调整或修改嘴唇的位置、形状或运动。此选项将覆盖动画区域设置，即动画区域将被忽略。'
        },
        'FaceExpressionRetargetingLipsMultiplierDecimalSlider': {
            'level': 3,
            'label': '重定向嘴唇倍数',
            'min_value': '0.00',
            'max_value': '2.00',
            'default': '1.00',
            'decimals': 2,
            'step': 0.01,
            'parentToggle': 'FaceExpressionRetargetingLipsEnableToggle & FaceExpressionEnableToggle',
            'requiredToggleValue': True,
            'help': '重定向嘴唇的倍数值。'
        },
    },
}