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
from utils.dataloaders import (
    IMG_FORMATS,
    VID_FORMATS,
    LoadImages,
    LoadScreenshots,
    LoadStreams,
)
from utils.general import (
    check_file,
    check_img_size,
    non_max_suppression,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
    LOGGER,
)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run_detect(
    weights,  # model path(s)
    source,  # file/dir/URL/glob/screen/0(webcam)
    data=None,  # dataset.yaml path, can be None for detection only
    imgsz=640,  # inference size (pixels)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IoU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    sort_boxes=False,  # sort boxes by center if True
):
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".txt") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    # Run inference
    model.warmup(imgsz=(1 if pt else len(dataset), 3, *imgsz))  # warmup
    results = []
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred[0][1], conf_thres, iou_thres, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            p, im0 = path, im0s
            if webcam:  # batch_size >= 1
                p, im0 = path[i], im0s[i]

            p = Path(p)  # to Path
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy()  # for save_crop

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    results.append(
                        {
                            "path": str(p),
                            "bbox": [int(x) for x in xyxy],
                            "conf": float(conf),
                            "cls": int(cls),
                        }
                    )

    if sort_boxes:
        results = sorted(
            results,
            key=lambda x: (
                (x["bbox"][0] + x["bbox"][2]) / 2,
                (x["bbox"][1] + x["bbox"][3]) / 2,
            ),
        )

    return results


def detect_image_dual(
    weights,
    source,
    data=None,  # make data optional
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device="",
    half=False,
    dnn=False,
    sort_boxes=False,
):
    return run_detect(
        weights=weights,
        source=source,
        data=data,
        imgsz=imgsz,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=max_det,
        device=device,
        half=half,
        dnn=dnn,
        sort_boxes=sort_boxes,
    )
