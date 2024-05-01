from models import *

def return_fasterrcnn_resnet50_fpn(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_resnet50_fpn.create_model(
        num_classes, pretrained=pretrained, coco_model=coco_model
    )
    return model

def return_fasterrcnn_resnet152(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_resnet152.create_model(
        num_classes, pretrained, coco_model
    )
    return model

def return_fasterrcnn_resnet50_fpn_v2(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_resnet50_fpn_v2.create_model(
        num_classes, pretrained=pretrained, coco_model=coco_model
    )
    return model

create_model = {
    'fasterrcnn_resnet50_fpn': return_fasterrcnn_resnet50_fpn,
    'fasterrcnn_resnet152': return_fasterrcnn_resnet152,
    'fasterrcnn_resnet50_fpn_v2': return_fasterrcnn_resnet50_fpn_v2
}