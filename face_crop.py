#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crop out the face
"""

import os
from math import floor
import cv2

INPUT_DIR = 'input'
OUTPUT_DIR = 'raw_input'
face_casc = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')


def crop(image, rect, dev=False):
    """Crop an image (img) from rect, returns the result image"""
    xb, yb, w, h = rect
    xe = xb + w
    ye = yb + h
    if dev:
        dev = floor(0.075 * ((w * h) ** 0.5))
        ye += dev
    return image[yb: ye, xb: xe]


if __name__ == '__main__':
    os.mkdir('raw_input')
    os.mkdir('standard_input')
    os.mkdir('output')
    for file in os.listdir(INPUT_DIR):
        img = cv2.imread(os.path.join(INPUT_DIR, file))
        face_rect = face_casc.detectMultiScale(
            img,
            scaleFactor=1.1,
            minNeighbors=10,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        img = crop(img, face_rect[0], True)
        cv2.imwrite(os.path.join(OUTPUT_DIR, file), img)
