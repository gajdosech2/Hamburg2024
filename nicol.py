import cv2
import numpy as np
from ultralytics import YOLO
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS


ROOT_DIR = "/home/g/gajdosech2/Hamburg2024/"
CLASS_NAMES = ['shot_glass', 'whisky_glass', 'water_glass', 'beer_glass', 'wine_glass', 'high_glass']
image = cv2.resize(mmcv.imread(ROOT_DIR + 'sample.png', channel_order='rgb'), (640, 360))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Our pre-trained RTMDet Model
rtmdet_model = init_detector(ROOT_DIR + 'config.py', ROOT_DIR + 'checkpoint.pth', device='cpu') 
visualizer = VISUALIZERS.build(rtmdet_model.cfg.visualizer)
visualizer.dataset_meta = rtmdet_model.dataset_meta

rtmdet_result = inference_detector(rtmdet_model, image)

#heatmap = rtmdet_result.pred_instances.heatmap.detach().cpu().numpy()[0]
#heatmap_normalized = cv2.normalize((heatmap * 255).astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
#heatmap_normalized = heatmap_normalized.astype(np.uint8)
#heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
#cv2.imwrite(f"heatmap.png", heatmap_colored)

print(rtmdet_result.pred_instances.bboxes.detach().cpu().numpy())
print([CLASS_NAMES[i] for i in rtmdet_result.pred_instances.labels.detach().cpu().numpy()])
print(rtmdet_result.pred_instances.scores.detach().cpu().numpy())
visualizer.add_datasample('result', image, data_sample=rtmdet_result, draw_gt = False, wait_time=0, pred_score_thr=0.1)
cv2.imwrite('prediction_rtmdet.png', visualizer.get_image())


# YOLO-World Model
yolo_model = YOLO("yolov8l-worldv2.pt")  
yolo_model.set_classes(CLASS_NAMES)

yolo_result = yolo_model.predict(image, conf=0.01)[0]
print(yolo_result.boxes.xyxy.detach().cpu().numpy())
print([CLASS_NAMES[int(i)] for i in yolo_result.boxes.cls.detach().cpu().numpy()])
print(yolo_result.boxes.conf.detach().cpu().numpy())
yolo_result.save("prediction_yolo.png")



