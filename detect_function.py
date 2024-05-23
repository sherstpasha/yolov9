import os
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadCV2Image
from utils.general import (Profile, check_img_size, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox


import numpy as np
import cv2

def sort_boxes_by_center(boxes):
    def center(box):
        xmin, ymin, xmax, ymax = box['bbox']
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        return center_y, center_x
    return sorted(boxes, key=center)

@smart_inference_mode()
def run(
        weights=Path('C:/Users/user/Desktop/yolov9/yolo.pt'),  # model path or triton URL
        source=None,  # file/dir/URL/glob/screen/0(webcam) or numpy image
        data=Path('C:/Users/user/Desktop/yolov9/data/coco.yaml'),  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference,
        sort_boxes=False  # sort boxes by center if True
):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    if isinstance(source, np.ndarray):
        dataset = LoadCV2Image(source, img_size=imgsz, stride=stride, auto=True)
    else:
        source = str(source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    model.warmup(imgsz=(1 if pt or model.triton else 1, 3, *imgsz))  # warmup
    results = []
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = model(im, augment=augment, visualize=visualize)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for det in pred:  # per image
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    results.append({
                        "path": path,
                        "bbox": [int(x) for x in xyxy],
                        "conf": float(conf),
                        "cls": int(cls)
                    })

    if sort_boxes:
        results = sort_boxes_by_center(results)

    return results


def detect_image(weights, source, data=None, imgsz=(640, 640), conf_thres=0.25, iou_thres=0.45, max_det=1000, device='', classes=None, agnostic_nms=False, augment=False, visualize=False, half=False, dnn=False, sort_boxes=False):
    return run(
        weights=weights,
        source=source,
        data=data,
        imgsz=imgsz,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=max_det,
        device=device,
        classes=classes,
        agnostic_nms=agnostic_nms,
        augment=augment,
        visualize=visualize,
        half=half,
        dnn=dnn,
        sort_boxes=sort_boxes
    )