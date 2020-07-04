from detectron2.config import CfgNode


def add_retinanet_deploy_config(cfg: CfgNode):
    _C = cfg

    _C.MODEL.DEPLOY = CfgNode()

    # Input shape for the deployed model.
    _C.MODEL.DEPLOY.INPUT_SHAPE = (1, 3, 540, 960)
    # ONNX opset version.
    _C.MODEL.DEPLOY.OPSET_VERSION = 9
    # Path to the exported ONNX model file.
    _C.MODEL.DEPLOY.ONNX_PATH = "models/retinanet.onnx"
    # Path to the serialized TensorRT engine file.
    _C.MODEL.DEPLOY.ENGINE_PATH = "models/retinanet.engine"
    # Path to the exported anchors.
    _C.MODEL.DEPLOY.ANCHORS_PATH = "models/retinanet_anchors.pth"
    # Maximum batch size for the TensorRT engine.
    _C.MODEL.DEPLOY.MAX_BATCH_SIZE = 1
    # Maximum workspace size for the TensorRT engine.
    _C.MODEL.DEPLOY.MAX_WORKSPACE_SIZE = 1 << 25
    # Whether to use FP16 precision.
    _C.MODEL.DEPLOY.FP16_MODE = True
    # Whether to force rebuilding the TensorRT engine.
    _C.MODEL.DEPLOY.FORCE_REBUILD = False
