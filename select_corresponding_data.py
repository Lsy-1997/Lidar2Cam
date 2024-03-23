import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
import yaml
import datetime
import glob
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

def select_corresponding_data(img_dir, pose_dir, reference_img_dir):

    img_ext = '.jpg'
    pose_ext = '.pose'
    images_file = []
    for file_path in glob.glob(os.path.join(img_dir, f'*{img_ext}')):
            images_file.append(os.path.splitext(os.path.basename(file_path))[0])

    poses_file = []
    for file_path in glob.glob(os.path.join(pose_dir, f'*{pose_ext}')):
        poses_file.append(os.path.splitext(os.path.basename(file_path))[0])

    reference_images_file = []
    for file_path in glob.glob(os.path.join(reference_img_dir, f'*{img_ext}')):
            reference_images_file.append(os.path.splitext(os.path.basename(file_path))[0])

    images_file.sort(key=lambda x: int(x))
    poses_file.sort(key=lambda x: int(x))

    # 点云快图像1s，消除时间差
    del images_file[0]
    del poses_file[-1]
    reference_images_file.sort(key=lambda x: int(x))

    root_dir = os.path.dirname(img_dir)

    dst_img_dir = os.path.join(root_dir, 'selected_image')
    dst_pose_dir = os.path.join(root_dir, 'selected_pose')

    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_pose_dir, exist_ok=True)

    # show_img_dir = os.path.join(root_dir, 'drawed_image_10m_-1.2-0_conv-method')

    cnt = 0
    for file in reference_images_file:
        if file in images_file:
            cnt = cnt + 1
            index = images_file.index(file)
            pose_file_name = poses_file[index]

            image_path = os.path.join(img_dir, file + img_ext)
            pose_path = os.path.join(pose_dir, pose_file_name + pose_ext)

            dst_image_path = os.path.join(dst_img_dir, file + img_ext)
            dst_pose_path = os.path.join(dst_pose_dir, pose_file_name + pose_ext)

            # 查看当初选择标注的数据，在建好图的点云投影下是什么样子
            # show_image_path = os.path.join(show_img_dir, file + img_ext)
            # if os.path.exists(show_image_path):
            #     image = cv2.imread(show_image_path)
            #     cv2.imshow(f'Image: {file+img_ext}', image)
            #     cv2.waitKey(0)
            #     # image = Image.open(show_image_path)

            #     # plt.imshow(image)
            #     # plt.show()
            # else :
            #     print('show image doesn\'t exitst')

            shutil.copy(image_path, dst_image_path)
            shutil.copy(pose_path, dst_pose_path)

if __name__ == '__main__':
    imgs_dir = 'correspond_data/after_mapping/in_1221_2023-12-21-16-06-08/image'
    poses_dir = 'correspond_data/after_mapping/in_1221_2023-12-21-16-06-08/pointcloud'
    reference_img_dir = 'correspond_data/in_1221_2023-12-21-16-06-08/selected_image'
    select_corresponding_data(imgs_dir, poses_dir, reference_img_dir)