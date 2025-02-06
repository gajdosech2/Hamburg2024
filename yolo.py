import os
import json
import shutil


def convert_coco_to_yolo_detection(coco_json_path, images_dir, output_dir):
    """Converts COCO JSON dataset to YOLO detection format (bounding boxes)."""
    
    output_images_dir = os.path.join(output_dir, "images")
    output_labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # Load COCO JSON
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    images = {img["id"]: img for img in coco_data["images"]}
    annotations = coco_data["annotations"]
    category_map = {cat["id"]: idx for idx, cat in enumerate(coco_data["categories"])}

    for img in images.values():
        old_path = img["file_name"]
        new_filename = old_path.replace("/", "_")
        new_image_path = os.path.join(output_images_dir, new_filename)

        old_image_full_path = os.path.join(images_dir, old_path)
        if os.path.exists(old_image_full_path):
            shutil.copy(old_image_full_path, new_image_path)
        else:
            print(f"Warning: {old_image_full_path} not found!")

        label_file = os.path.join(output_labels_dir, new_filename.replace(".png", ".txt"))
        with open(label_file, "w") as lf:
            for ann in annotations:
                if ann["image_id"] == img["id"]:
                    category_id = category_map[ann["category_id"]]
                    x, y, w, h = ann["bbox"]
                    x_center = (x + w / 2) / img["width"]
                    y_center = (y + h / 2) / img["height"]
                    w /= img["width"]
                    h /= img["height"]
                    lf.write(f"{category_id} {x_center} {y_center} {w} {h}\n")

    print("YOLO detection conversion complete!")


def convert_coco_to_yolo_segmentation(coco_json_path, images_dir, output_dir):
    """Converts COCO JSON dataset to YOLO segmentation format (polygon masks)."""
    
    output_images_dir = os.path.join(output_dir, "images")
    output_labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # Load COCO JSON
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    images = {img["id"]: img for img in coco_data["images"]}
    annotations = coco_data["annotations"]
    category_map = {cat["id"]: idx for idx, cat in enumerate(coco_data["categories"])}

    for img in images.values():
        old_path = img["file_name"]
        new_filename = old_path.replace("/", "_")
        new_image_path = os.path.join(output_images_dir, new_filename)

        old_image_full_path = os.path.join(images_dir, old_path)
        if os.path.exists(old_image_full_path):
            shutil.copy(old_image_full_path, new_image_path)
        else:
            print(f"Warning: {old_image_full_path} not found!")

        label_file = os.path.join(output_labels_dir, new_filename.replace(".png", ".txt"))
        with open(label_file, "w") as lf:
            for ann in annotations:
                if ann["image_id"] == img["id"]:
                    category_id = category_map[ann["category_id"]]
                    segmentation = ann["segmentation"][0]  # Single segmentation list
                    normalized_segmentation = [
                        (segmentation[i] / img["width"], segmentation[i + 1] / img["height"])
                        for i in range(0, len(segmentation), 2)
                    ]
                    segmentation_str = " ".join(f"{x} {y}" for x, y in normalized_segmentation)
                    lf.write(f"{category_id} {segmentation_str}\n")

    print("YOLO segmentation conversion complete!")




coco_json_path = "/home/g/gajdosech2/Hamburg2024/coco_annotations_val.json"  # Change to actual path
images_dir = "/home/g/gajdosech2/Hamburg2024"  # Base directory of images
output_dir = "/home/g/gajdosech2/Hamburg2024/yolo_dataset"  # Output directory

convert_coco_to_yolo_segmentation(coco_json_path, images_dir, output_dir)