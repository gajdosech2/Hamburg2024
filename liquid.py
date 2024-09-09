import cv2
import numpy as np

import os

ROOT_DIR = "/home/g/gajdosech2/"
#ROOT_DIR = "/export/home/gajdosec/"

os.chdir(ROOT_DIR + "/Hamburg2024")

# Define HSV ranges for the liquids (you may need to tune these)
color_ranges = {
    'cola': ([0, 50, 50], [10, 255, 255]),  # dark brown
    'wine': ([160, 100, 100], [180, 255, 255]),  # red
    'beer': ([10, 50, 50], [50, 255, 255]),  # orange
}

def filter_depth(rgb_frame, depth_frame, max_distance_meters=0.7):
    """ Apply a depth filter to only keep objects within a certain range. """
    depth_mask = depth_frame <= max_distance_meters * 1000  # Convert to mm
    return rgb_frame * depth_mask[:, :, np.newaxis]

def color_segmentation(hsv_frame, color_ranges):
    """ Apply color segmentation to find liquids based on HSV thresholds. """
    masks = {}
    for color_name, (lower, upper) in color_ranges.items():
        lower_bound = np.array(lower)
        upper_bound = np.array(upper)
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        masks[color_name] = mask
    return masks

def postprocess_and_find_contours(mask):
    """ Post-process masks and find contours. """
    # Apply some morphology to clean the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours of the objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def classify_glass(contour):
    """ Heuristic-based glass classification based on contour shape. """
    # You can calculate shape features like aspect ratio, circularity, etc.
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    
    # Classify based on aspect ratio
    if aspect_ratio < 0.6:  # Tall and thin - could be wine glass
        return "wine_glass"
    elif aspect_ratio > 0.9:  # Short and wide - could be beer glass
        return "beer_glass"
    else:  # In between - could be shot glass or other
        return "shot_glass"

# Load saved images
rgb_image = cv2.imread('data/9.png')  # Replace with the actual path to your RGB image
depth_image = np.load('data/9.npy')  # Replace with the actual path to your depth image (saved as a NumPy array)

# 1. Filter based on depth
filtered_image = filter_depth(rgb_image, depth_image)

# 2. Convert to HSV and perform color segmentation
hsv_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2HSV)
masks = color_segmentation(hsv_image, color_ranges)

for color_name, mask in masks.items():
    # 3. Post-process and find contours
    contours = postprocess_and_find_contours(mask)
    
    for contour in contours:
        # 4. Classify the type of glass
        glass_type = classify_glass(contour)
        
        # Draw bounding boxes and labels
        x, y, w, h = cv2.boundingRect(contour)
        if w < 100 or w < 100 or w > 300 or h > 300:
            continue
        cv2.rectangle(filtered_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(filtered_image, f'{color_name} - {glass_type}', 
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Show the processed image
cv2.imwrite('work_dirs/detected.png', filtered_image)
