import mmcv
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from mmengine import Config
import os
os.chdir("/export/home/gajdosec/Hamburg2024")

# Path to load the COCO annotation file
annotation_file = './ballondatasets/balloon/train/annotation_coco.json'

# Initialise the COCO object
coco = COCO(annotation_file)

# Get all category tags and corresponding category IDs
categories = coco.loadCats(coco.getCatIds())
category_id_to_name = {cat['id']: cat['name'] for cat in categories}

# Print all category IDs and corresponding category names
for category_id, category_name in category_id_to_name.items():
    print(f"Category ID: {category_id}, Category Name: {category_name}")


cfg = Config.fromfile('./configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco.py')

