#!/usr/bin/env python3
# -*- coding : utf-8 -*-

"""A facial comparison tool
Principle:
Find the coordinates of some feature points
    - we choose the leftmost point of mouth M and 6 points of the right eye P1~P6.
Calculate the average vector of vectors starting from M pointing to P1~P6.
Use the formula k = {[|a| - abs(|a| - |b|)] / (|a| * |b|)} * cos<a, b>
Compare the `k`s of pairs of images.
"""

import os.path
from math import sqrt

import cv2
import dlib
import numpy as np

BEG_PTS = list(range(36, 48))
FIN_PTS = list(range(48, 61))
MODEL_PATH = 'model/shape_predictor_68_face_landmarks.dat'
INPUT_DIR = 'standard_input'
OUTPUT_DIR = 'output'


class _Vector:
    """A vector class, used for calculation."""

    def __init__(self, orig=(0, 0)):
        if isinstance(orig, dlib.point) or isinstance(orig, dlib.vector):
            self.x = orig.x
            self.y = orig.y
        else:
            self.x = orig[0]
            self.y = orig[1]

    def norm(self):
        return sqrt(self.x ** 2 + self.y ** 2)

    def __mul__(self, other):
        if isinstance(other, _Vector):
            return self.x * other.x + self.y * other.y
        else:
            return _Vector([self.x * other, self.y * other])

    def __add__(self, other):
        return _Vector([self.x + other.x, self.y + other.y])

    def __sub__(self, other):
        return _Vector([self.x - other.x, self.y - other.y])

    def __truediv__(self, other):
        return _Vector([self.x / other, self.y / other])

    def __str__(self):
        return f'({self.x}, {self.y})'

    def cos(self, other):
        return self * other / (self.norm() * other.norm())

    def adj(self):
        return _Vector([self.x, -self.y])


def k(v1: _Vector, v2: _Vector):
    return (v1.norm() - abs(v1.norm() - v2.norm())) / v1.norm() * v1.cos(v2)


def dist(v1: _Vector, v2: _Vector):
    return sqrt((v1.x - v2.x) ** 2 + (v1.y - v2.y) ** 2)


def gen_vec(par_img: np.ndarray, save_fn: str = ''):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(MODEL_PATH)
    face = detector(par_img, 1)[0]
    shape = predictor(par_img, face)
    vec_sum = _Vector([0, 0])

    with open(os.path.join(OUTPUT_DIR, save_fn), 'w') as f:
        f.write('Starting Points: \n')
        for pt in BEG_PTS:
            cv2.circle(par_img, (shape.parts()[pt].x, shape.parts()[pt].y), 3, (25, 100, 50), 2)
            cv2.putText(par_img, str(pt), (shape.parts()[pt].x, shape.parts()[pt].y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (25, 100, 50), 2)
            f.write(f'\t{pt}: ({shape.parts()[pt].x}, {shape.parts()[pt].y})\n')
        f.write('Ending Points: \n')
        for pt in FIN_PTS:
            cv2.circle(par_img, (shape.parts()[pt].x, shape.parts()[pt].y), 3, (25, 0, 50), 2)
            cv2.putText(par_img, str(pt), (shape.parts()[pt].x, shape.parts()[pt].y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (25, 0, 50), 2)
            f.write(f'\t{pt}: ({shape.parts()[pt].x}, {shape.parts()[pt].y})\n')
    for beg in BEG_PTS:
        for fin in FIN_PTS:
            vec_beg = _Vector(shape.parts()[beg]).adj()
            vec_fin = _Vector(shape.parts()[fin]).adj()
            vector = vec_fin - vec_beg
            vec_sum = vec_sum + vector
    vec_avg = vec_sum / (len(BEG_PTS) * len(FIN_PTS))
    return vec_avg, par_img


if __name__ == '__main__':
    fn_def = input('Select an image to compare with: ')
    if not os.path.exists(os.path.join(INPUT_DIR, fn_def)):
        raise SystemExit('The input image does not exist!')
    img_def = cv2.imread(os.path.join(INPUT_DIR, fn_def))
    vec_def, p_img = gen_vec(img_def, fn_def + '.txt')
    print(vec_def)
    cv2.imwrite(os.path.join(OUTPUT_DIR, fn_def), p_img)

    try:
        while True:
            fn = input('Enter file name: ')
            if not os.path.exists(os.path.join(INPUT_DIR, fn)):
                raise FileNotFoundError
            img = cv2.imread(os.path.join(INPUT_DIR, fn))
            vec, p_img = gen_vec(img, fn + '.txt')
            cv2.imwrite(os.path.join(OUTPUT_DIR, fn), p_img)
            print('Vector:', vec)
            print('k =', k(vec, vec_def))
    except KeyboardInterrupt:
        raise SystemExit('Halt.')
    except FileNotFoundError:
        raise SystemExit('File not found.')
