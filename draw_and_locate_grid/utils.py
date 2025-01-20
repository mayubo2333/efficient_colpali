import os
import cv2
import base64
from math import ceil
from PIL import Image
from uuid import uuid4

import sys
sys.path.append("/mnt/petrelfs/mayubo/colpali")
from colpali_engine.models.qwen2.colqwen2.processing_colqwen2 import ColQwen2Processor


def draw_grid(image_path, grid_size, new_image_path):
    """
    Draw grid lines on the image.
    """
    image = cv2.imread(image_path)
    new_height, new_width = ColQwen2Processor.smart_resize_helper(
        height=image.shape[0],
        width=image.shape[1], 
        factor=28,
        max_ratio=200, 
        min_pixels=4*28*28,
        max_pixels=768*28*28
    )
    # resize the image to the required size
    image = cv2.resize(image, (new_width, new_height))
    for coor_id in range(grid_size, max(new_height, new_width), grid_size):
        cv2.line(image, (coor_id, 0), (coor_id, image.shape[0]), (0, 255, 0), 2)
        cv2.line(image, (0, coor_id), (image.shape[1], coor_id), (0, 255, 0), 2)

    # Add sequential number (0, 1, .....) at the top-left of each grid
    height_num, width_num = ceil(new_height/grid_size), ceil(new_width/grid_size)
    for i in range(height_num):
        for j in range(width_num):
            x, y = j*grid_size, i*grid_size
            cv2.putText(image, str(i*width_num+j), (x+20, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imwrite(new_image_path, image)
    return new_height, new_width


def encode_image_to_base64(img):
    if isinstance(img, str):
        img = Image.open(img)
    
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    tmp = os.path.join('/tmp', str(uuid4()) + '.jpg')
    img.save(tmp)
    with open(tmp, 'rb') as image_file:
        image_data = image_file.read()
    ret = base64.b64encode(image_data).decode('utf-8')
    os.remove(tmp)
    return ret