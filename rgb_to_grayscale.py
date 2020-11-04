import numpy as np
from PIL import Image
import os

path = 'Data/img_train_skeleton'
destination_path = 'Data/img_train_skeleton_grayscale'
os.mkdir(destination_path)

for pic in os.listdir(path):
    if pic.endswith('.png'):
        pic_converted = Image.open(os.path.join(path, pic), 'r')
        pic_converted = pic_converted.convert('L')
        pic_converted.save(os.path.join(destination_path, pic))
