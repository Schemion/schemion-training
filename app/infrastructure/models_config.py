FASTERRCNN_ARCHITECTURES = {
    "resnet50_fpn": "fasterrcnn_resnet50_fpn",
    "resnet50_fpn_v2": "fasterrcnn_resnet50_fpn_v2",
    "mobilenet_v3_large_fpn": "fasterrcnn_mobilenet_v3_large_fpn",
    "mobilenet_v3_large_320_fpn": "fasterrcnn_mobilenet_v3_large_320_fpn",
}

FASTERRCNN_ALIASES = {
    "fasterrcnn": "faster_rcnn",
    "frcnn": "faster_rcnn",
    "faster_rcnn": "faster_rcnn",
    "faster_r_cnn":"faster_rcnn"
}

# больше в голову не пришло
ARCHITECTURE_ALIASES = {
    "resnet50": "resnet50_fpn",
    "resnet50v2": "resnet50_fpn_v2",
    "mobilenet": "mobilenet_v3_large_fpn",
    "mobilenet_320": "mobilenet_v3_large_320_fpn",
    "fpn": "resnet50_fpn",
    "fpnv2": "resnet50_fpn_v2",
}
