import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def preprocessing(img, hr_res):
    img = tf.random

class DataLoader():
    def __init__(self, name, img_res=(224,224,3)):
        self.name = name
        self.img_res = img_res
        self.path = "./dataset/{}/hr/".format(self.name)
        self.ds = sorted(os.listdir(self.path))
        self.shuffle_buffer_size = 32

    def generate_img(self):
        for img in self.ds:
            yield cv2.imread(self.path + img)           

    def load_data(self, batch_size=4):
        ds = tf.data.Dataset.from_generator(self.generate_img, output_types=tf.float32)
        ds = ds.map(self.preprocess)
        ds = ds.shuffle(self.shuffle_buffer_size)
        ds = ds.prefetch(buffer_size=2)
        ds = ds.batch(batch_size)
        return ds

    def preprocess(self, img):
        h, w, _ = self.img_res
        low_h, low_w = int(h / 4), int(w / 4)
        img_patch_hr = tf.image.random_crop(img, [h,w,3], seed=1)
        # img_patch_hr = tf.cast(img_patch_hr, tf.float32)
        img_patch_lr = tf.image.resize(img_patch_hr, (low_h,low_w))
        img_patch_lr = (img_patch_lr-127.5)/127.5
        # img_patch_lr = tf.cast(img_patch_lr, tf.uint8)
        img_patch_hr = (img_patch_hr-127.5)/127.5
        # img_patch_hr = tf.cast(img_patch_hr, tf.uint8)
        return img_patch_lr, img_patch_hr
