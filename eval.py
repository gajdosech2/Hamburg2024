import json
import cv2
import numpy as np
import mmcv
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS


ROOT_DIR = "/home/g/gajdosech2/Hamburg2024/"
CLASS_NAMES = ['shot_glass', 'whisky_glass', 'water_glass', 'beer_glass', 'wine_glass', 'high_glass']


def load_ground_truth(json_path):
    return COCO(json_path)


def extract_bboxes(gt_json, pd_json, image_id):
    with open(gt_json, "r") as f:
        gt_data = json.load(f)

    with open(pd_json, "r") as f:
        pd_data = json.load(f)

    gt_annotations = []
    for ann in gt_data["annotations"]:
        if ann["image_id"] == image_id:
            bbox = ann["bbox"] 
            gt_annotations.append(bbox)

    pd_annotations = []
    for ann in pd_data:
        if ann["image_id"] == image_id:
            bbox = ann["bbox"] 
            pd_annotations.append(bbox)

    return gt_annotations, pd_annotations


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def average_iou(gt_boxes, pred_boxes, thresh=0.5):
    total_iou = 0
    matched = 0

    for gt_box in gt_boxes:
        best_iou = 0
        for pred_box in pred_boxes:
            iou = compute_iou(gt_box, pred_box)
            if iou > thresh: 
                best_iou = max(best_iou, iou)

        if best_iou > 0:
            total_iou += best_iou
            matched += 1

    return total_iou / matched if matched > 0 else 0


def convert_predictions_to_coco_format(predictions, image_id):
    results = []
    for bbox, label, score in predictions:
        x, y, x2, y2 = bbox
        w, h = x2 - x, y2 - y
        results.append({
            "image_id": image_id,
            "category_id": label + 1, 
            "bbox": [x * 2, y * 2, w * 2, h * 2],
            "score": score  # Confidence score (important for AP calculation)
        })
    return results


def prediction_to_coco(json_path):
    coco_gt = load_ground_truth(json_path)
    image_ids = coco_gt.getImgIds()
    coco_results = {model_name: [] for model_name in ["yolo", "rtmdet"]}

    yolo_model = YOLO("yolov8l-worldv2.pt")  
    yolo_model.set_classes(CLASS_NAMES)
    rtmdet_model = init_detector(ROOT_DIR + 'config.py', ROOT_DIR + 'checkpoint_data3_nokp.pth', device='cpu') 

    visualizer = VISUALIZERS.build(rtmdet_model.cfg.visualizer)
    visualizer.dataset_meta = rtmdet_model.dataset_meta

    for image_id in image_ids:
        img_info = coco_gt.loadImgs(image_id)[0]
        image_path = img_info["file_name"]

        image = cv2.resize(mmcv.imread(ROOT_DIR + image_path, channel_order='rgb'), (640, 360))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        yolo_result = yolo_model.predict(image, conf=0.01)[0]
        yolo_bboxes = yolo_result.boxes.xyxy.detach().cpu().numpy().astype(int).tolist()
        yolo_classes = yolo_result.boxes.cls.detach().cpu().numpy().astype(int).tolist()
        yolo_scores = yolo_result.boxes.conf.detach().cpu().numpy().astype(float).tolist()
        yolo_predictions = list(zip(yolo_bboxes, yolo_classes, yolo_scores))
        coco_results["yolo"].extend(convert_predictions_to_coco_format(yolo_predictions, image_id))
        yolo_result.save("prediction_eval_yolo.png")
 
        rtmdet_result = inference_detector(rtmdet_model, image)
        rtmdet_bboxes = rtmdet_result.pred_instances.bboxes.detach().cpu().numpy().astype(int).tolist()
        rtmdet_classes = rtmdet_result.pred_instances.labels.detach().cpu().numpy().astype(int).tolist()
        rtmdet_scores = rtmdet_result.pred_instances.scores.detach().cpu().numpy().astype(float).tolist()
        rtmdet_predictions = list(zip(rtmdet_bboxes, rtmdet_classes, rtmdet_scores))
        coco_results["rtmdet"].extend(convert_predictions_to_coco_format(rtmdet_predictions, image_id))
        visualizer.add_datasample('result', image, data_sample=rtmdet_result, draw_gt=False, wait_time=0, pred_score_thr=0.2)
        cv2.imwrite('prediction_eval_rtmdet.png', visualizer.get_image())



    for model_name, results in coco_results.items():
        with open(f"{model_name}_results.json", "w") as f:
            json.dump(results, f)


def evaluate_models(json_path, models):
    coco_gt = load_ground_truth(json_path)

    for model_name in models:
        #coco_gt.dataset["categories"] = [{"id": 1, "name": "object"}]  # Single category
        coco_dt = coco_gt.loadRes(f"{model_name}_results.json")

        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        #coco_eval.params.catIds = [4]  # Ignore class labels
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        ious = []
        for image_id in coco_gt.getImgIds():
            gt_boxes, pd_boxes = extract_bboxes(json_path, f"{model_name}_results.json", image_id)
            iou = average_iou(gt_boxes, pd_boxes)
            ious.append(iou)

        print(f"Average BBOX IoU for {model_name}: {np.mean(ious):.4f}")




if __name__ == "__main__":
    gt_json_path = "/home/g/gajdosech2/Hamburg2024/annotations/coco_annotations_val_data_3_frag_nokp.json" 
    #prediction_to_coco(gt_json_path)
    evaluate_models(gt_json_path, ["rtmdet", "yolo"])