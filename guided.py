import cv2
import requests
import numpy as np
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import torch


model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = model.to(torch.device("cuda"))
model.eval()

#https://iili.io/2ZIjon2.jpg
#https://iili.io/2ZIjTF9.jpg
#https://iili.io/2ZIjxGS.jpg
#https://iili.io/2ZIjz67.jpg
#https://iili.io/2ZIjuae.jpg
#https://iili.io/2ZIjA8u.jpg

# Input image
url = "https://iili.io/2ZIjuae.jpg"
image = Image.open(requests.get(url, stream=True).raw)
target_sizes = torch.Tensor([image.size[::-1]])

# Query image
query_url = "https://iili.io/2ZIjxGS.jpg"
query_image = Image.open(requests.get(query_url, stream=True).raw)

# Process input and query image
inputs = processor(images=image, query_images=query_image, return_tensors="pt").to(torch.device("cuda"))

# Print input names and shapes
for key, val in inputs.items():
    print(f"{key}: {val.shape}")

# Get predictions
with torch.no_grad():
  outputs = model.image_guided_detection(**inputs)

for k, val in outputs.items():
    if k not in {"text_model_output", "vision_model_output"}:
        print(f"{k}: shape of {val.shape}")

print("\nVision model outputs")
for k, val in outputs.vision_model_output.items():
    print(f"{k}: shape of {val.shape}")

img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
outputs.logits = outputs.logits.cpu()
outputs.target_pred_boxes = outputs.target_pred_boxes.cpu()

results = processor.post_process_image_guided_detection(outputs=outputs, threshold=0.7, nms_threshold=0.3, target_sizes=target_sizes)
boxes, scores = results[0]["boxes"], results[0]["scores"]

# Draw predicted bounding boxes
for box, score in zip(boxes, scores):
    box = [int(i) for i in box.tolist()]

    img = cv2.rectangle(img, box[:2], box[2:], (255,0,0), 5)
    if box[3] + 25 > 768:
        y = box[3] - 10
    else:
        y = box[3] + 25

cv2.imwrite("/home/g/gajdosech2/Hamburg2024/work_dirs/debug/owlvit.png", img[:,:,::-1])