import cv2
import numpy as np
from ultralytics import YOLO
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS


ROOT_DIR = "/home/g/gajdosech2/Hamburg2024/"
CLASS_NAMES = ['shot_glass', 'whisky_glass', 'water_glass', 'beer_glass', 'wine_glass', 'high_glass']
image = cv2.resize(mmcv.imread(ROOT_DIR + 'data/sample.png', channel_order='rgb'), (640, 360))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Our pre-trained RTMDet Model
heatmap = True
threshold = 0.35
rtmdet_model = init_detector(ROOT_DIR + 'heat.py', ROOT_DIR + 'checkpoint.pth', device='cpu') 
#rtmdet_model = init_detector(ROOT_DIR + 'config.py', ROOT_DIR + 'checkpoint_nokp.pth', device='cpu') 
visualizer = VISUALIZERS.build(rtmdet_model.cfg.visualizer)
visualizer.dataset_meta = rtmdet_model.dataset_meta

rtmdet_result = inference_detector(rtmdet_model, image)
if heatmap:
    bg = image.copy()
    heatmap = rtmdet_result.pred_instances.heatmap.detach().cpu().numpy()[0]
    maxima_coords = []
    for j, (xmin, ymin, xmax, ymax) in enumerate(rtmdet_result.pred_instances.bboxes.detach().cpu().numpy()):
        if rtmdet_result.pred_instances.scores[j] > threshold:
            xmin, ymin, xmax, ymax = map(lambda v: int(round(v)), [xmin, ymin, xmax, ymax])
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(xmax, heatmap.shape[1]), min(ymax, heatmap.shape[0])
            roi = heatmap[ymin:ymax, xmin:xmax]
            if roi.size > 0:
                max_idx = np.unravel_index(np.argmax(roi), roi.shape)
                max_y, max_x = max_idx 
                maxima_coords.append((xmin + max_x, ymin + max_y))  
                cv2.circle(bg, (xmin + max_x, ymin + max_y), radius=3, color=(0, 255, 0), thickness=-1)
    print(maxima_coords)
    cv2.imwrite('keypoints_rtmdet.png', bg)

print(rtmdet_result.pred_instances.bboxes.detach().cpu().numpy())
print([CLASS_NAMES[i] for i in rtmdet_result.pred_instances.labels.detach().cpu().numpy()])
print(rtmdet_result.pred_instances.scores.detach().cpu().numpy())
visualizer.add_datasample('result', image, data_sample=rtmdet_result, draw_gt=False, wait_time=0, pred_score_thr=threshold)
cv2.imwrite('prediction_rtmdet.png', visualizer.get_image())


# YOLO-World Model
yolo_model = YOLO("yolov8l-worldv2.pt")  
yolo_model.set_classes(CLASS_NAMES)

yolo_result = yolo_model.predict(image, conf=0.01)[0]
print(yolo_result.boxes.xyxy.detach().cpu().numpy())
print([CLASS_NAMES[int(i)] for i in yolo_result.boxes.cls.detach().cpu().numpy()])
print(yolo_result.boxes.conf.detach().cpu().numpy())
yolo_result.save("prediction_yolo.png")



