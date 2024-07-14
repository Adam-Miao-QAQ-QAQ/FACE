#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scale the image till it's area is equal to the standard.
"""
from PIL import Image
import os
from math import sqrt

INPUT_DIR = 'raw_input'
OUTPUT_DIR = 'standard_input'

if __name__ == '__main__':
    standard = input("Select an image to compare with:")
    img = Image.open(os.path.join(INPUT_DIR, standard))
    std_size = sqrt(img.size[0] * img.size[1])

    for file in os.listdir(INPUT_DIR):
        img = Image.open(os.path.join(INPUT_DIR, file))
        size = sqrt(img.size[0] * img.size[1])
        ratio = std_size / size
        img = img.resize((round(img.size[0] * ratio), round(img.size[1] * ratio)))
        img.save(os.path.join(OUTPUT_DIR, file))
