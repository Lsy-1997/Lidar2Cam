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

import proj_pcd2cam
import proj_mapped_pointcloud2cam

class ExtrinsicAdjuster:
    def __init__(self, window, img_dir, pose_dir, calib_path, pcd_path):
        self.window = window
        self.window.title("Lidar2Image Extrinsic Adjuster")

        # some paramerters
        # self.show_type = 'reflectance'
        self.show_type = 'z'
        self.img_ext = '.jpg'
        self.pose_ext = '.pose'
        self.images_file = []
        self.img_dir = img_dir
        self.pose_dir = pose_dir
        for file_path in glob.glob(os.path.join(img_dir, f'*{self.img_ext}')):
            self.images_file.append(os.path.splitext(os.path.basename(file_path))[0])

        self.poses_file = []
        for file_path in glob.glob(os.path.join(pose_dir, f'*{self.pose_ext}')):
            self.poses_file.append(os.path.splitext(os.path.basename(file_path))[0])

        self.images_file.sort(key=lambda x: int(x))
        self.poses_file.sort(key=lambda x: int(x))

        # 点云慢图像1s
        del self.images_file[0]
        del self.poses_file[-1]
        # del self.images_file[-1]
        # del self.poses_file[0]

        self.file_num = 1
        image_path = os.path.join(img_dir, self.images_file[self.file_num] + self.img_ext)
        pose_path = os.path.join(pose_dir, self.poses_file[self.file_num] + self.pose_ext)

        # Load the image using OpenCV
        self.original_image = cv2.imread(image_path)
        self.image = self.original_image.copy()
        self.pc = proj_mapped_pointcloud2cam.load_pcd_data(pcd_path)

        self.pose = proj_mapped_pointcloud2cam.get_pose_matrix(pose_path)
        self.IMG_H, self.IMG_W = self.image.shape[:2]

        def create_slider(label_text, from_, to_, command):
            frame = ttk.Frame(window)
            label = ttk.Label(frame, text=label_text)
            label.pack(side=tk.LEFT)
            scale = ttk.Scale(frame, from_=from_, to=to_, orient="horizontal", command=command)
            scale.set(0)
            scale.pack(side=tk.LEFT)
            frame.pack()
            return scale

        # Create sliders with labels for Alpha, Beta, Gamma, X, Y, Z
        self.alpha_scale = create_slider("Alpha", -5, 5, self.update_extrinsic)
        self.beta_scale = create_slider("Beta", -2, 2, self.update_extrinsic)
        self.gamma_scale = create_slider("Gamma", -2, 2, self.update_extrinsic)
        self.x_scale = create_slider("X", -1, 1, self.update_extrinsic)
        self.y_scale = create_slider("Y", -1, 1, self.update_extrinsic)
        self.z_scale = create_slider("Z", -1, 1, self.update_extrinsic)
        self.progress_bar = create_slider("Progress Bar", 0, len(self.images_file)-1, self.move_progress_bar)

        # Set default value of sliders to 1 (no change)
        self.alpha_scale.set(0)
        self.beta_scale.set(0)
        self.gamma_scale.set(0)
        self.x_scale.set(0)
        self.y_scale.set(0)
        self.z_scale.set(0)
        self.progress_bar.set(self.file_num)

        # Place the sliders
        self.alpha_scale.pack()
        self.beta_scale.pack()
        self.gamma_scale.pack()
        self.x_scale.pack()
        self.y_scale.pack()
        self.z_scale.pack()
        self.progress_bar.pack()

        # Create a label to display the image
        self.image_label = ttk.Label(window)
        self.image_label.pack()

        # Create a save button
        self.save_button = ttk.Button(window, text="Save Extrinsic", command=self.save_extrinsic)
        self.save_button.pack(side=tk.LEFT)
        self.refresh_button = ttk.Button(window, text="Refresh Parameter", command=self.refresh_extrinsic)
        self.refresh_button.pack(side=tk.LEFT)
        self.prev_button = ttk.Button(window, text="Prev Image", command=self.prev_img)
        self.prev_button.pack(side=tk.LEFT)
        self.next_button = ttk.Button(window, text="Next Image", command=self.next_img)
        self.next_button.pack(side=tk.LEFT)
        self.save_img_button = ttk.Button(window, text="Save Img & PC", command=self.save_img_pointcloud)
        self.save_img_button.pack(side=tk.LEFT)
        self.change_type_button = ttk.Button(window, text="Change PC Type", command=self.change_type)
        self.change_type_button.pack(side=tk.LEFT)
        self.automation_button = ttk.Button(window, text="Automation", command=self.automation)
        self.automation_button.pack(side=tk.LEFT)

        # Initialize the image on the label
        self.intrisic, self.extrinsic= proj_pcd2cam.get_calib_param(calib_path)
        
        self.new_extrinsic = self.extrinsic.copy()
        # self.pointcloud = proj_pcd2cam.load_pcd_data(pointcloud_path)
        
        # self.update_extrinsic()

    def automation(self):
        img_save_dir = os.path.join(os.path.dirname(self.img_dir), 'drawed_image_10m_-1.2-0_conv-method')
        os.makedirs(img_save_dir, exist_ok=True)
        for i in range(len(self.images_file)-1):
            print(f'processing number: {i}, total: {len(self.images_file)-1}')
            self.file_num = i
            self.save_drawed_img(save_dir=img_save_dir)

    def save_drawed_img(self, save_dir):

        image_path = os.path.join(self.img_dir, self.images_file[self.file_num] + self.img_ext)
        pose_path = os.path.join(self.pose_dir, self.poses_file[self.file_num] + self.pose_ext)
        # self.pointcloud = proj_pcd2cam.load_pcd_data(pointcloud_path)
        self.original_image = cv2.imread(image_path)
        self.image = self.original_image.copy()
        self.pose = proj_mapped_pointcloud2cam.get_pose_matrix(pose_path)

        image_save_path = os.path.join(save_dir, self.images_file[self.file_num] + self.img_ext)
        self.update_extrinsic(image_save_path=image_save_path)

    def change_type(self):
        if self.show_type == 'z':
            self.show_type = 'reflectance'
        elif self.show_type == 'reflectance':
            self.show_type = 'z'

    def save_img_pointcloud(self):
        img_save_dir = os.path.join(os.path.dirname(self.img_dir), 'selected_image')
        pc_save_dir = os.path.join(os.path.dirname(self.pose_dir), 'selected_pointcloud')
        os.makedirs(img_save_dir, exist_ok=True)
        os.makedirs(pc_save_dir, exist_ok=True)

        image_path = os.path.join(self.img_dir, self.images_file[self.file_num] + self.img_ext)
        pointcloud_path = os.path.join(self.pose_dir, self.pose_file[self.file_num] + self.pc_ext)

        image_save_path = os.path.join(img_save_dir, self.images_file[self.file_num] + self.img_ext)
        pc_save_path = os.path.join(pc_save_dir, self.pose_file[self.file_num] + self.pc_ext)

        shutil.copy(image_path, image_save_path)
        shutil.copy(pointcloud_path, pc_save_path)

    def next_img(self):
        if(self.file_num+1<len(self.images_file)):
            self.file_num += 1
        # self.refresh_image()
        self.progress_bar.set(self.file_num)

    def prev_img(self):
        if(self.file_num-1>=0):
            self.file_num -= 1
        # self.refresh_image()
        self.progress_bar.set(self.file_num)

    def move_progress_bar(self, event=None):
        self.file_num = int(self.progress_bar.get())
        self.refresh_image()

    def refresh_image(self):
        image_path = os.path.join(self.img_dir, self.images_file[self.file_num] + self.img_ext)
        pose_path = os.path.join(self.pose_dir, self.poses_file[self.file_num] + self.pose_ext)
        # self.pointcloud = proj_pcd2cam.load_pcd_data(pointcloud_path)
        self.original_image = cv2.imread(image_path)
        self.image = self.original_image.copy()
        self.pose = proj_mapped_pointcloud2cam.get_pose_matrix(pose_path)

        self.update_extrinsic()

    def refresh_extrinsic(self):
        self.alpha_scale.set(0)
        self.beta_scale.set(0)
        self.gamma_scale.set(0)
        self.x_scale.set(0)
        self.y_scale.set(0)
        self.z_scale.set(0)
        self.update_extrinsic()
    
    def save_extrinsic(self):
        autoware_extrinsic = proj_pcd2cam.transform_from_autoware_to_normal(self.new_extrinsic)
        data_to_save = {
            "CameraExtrinsicMat": {
                "rows": 4,
                "cols": 4,
                "dt": "d",  # 数据类型, 这里假设是 'd' (double)
                "data": self.new_extrinsic.tolist(), 
            }
        }
        current_time = datetime.datetime.now()
        # 将数组写入 YAML 文件
        with open(f'{current_time}extrinsic.yaml', 'w') as file:
            yaml.dump(data_to_save, file)
        print("Extrinsic save successfully!")

    def update_extrinsic(self, image_save_path='', _=None):
        # Adjust the rotation parameters
        alpha, beta, gamma = self.alpha_scale.get(), self.beta_scale.get(), self.gamma_scale.get()
        x, y, z = self.x_scale.get(), self.y_scale.get(), self.z_scale.get()
        
        alpha = (alpha / 3.14) * (1 / 360.0)
        beta = (beta / 3.14) * (1 / 360.0)
        gamma = (gamma / 3.14) * (1 / 360.0)
        # 3个旋转矩阵
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)]
        ])
        Ry = np.array([
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)]
        ])
        Rz = np.array([
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma),  np.cos(gamma), 0],
            [0, 0, 1]
        ])
        rotation_mat = self.extrinsic[:3,:3]
        rotation_mat = Rx.dot(Ry.dot(Rz.dot(rotation_mat)))
        self.new_extrinsic[:3, :3] = rotation_mat
        self.new_extrinsic[0, 3] = self.extrinsic[0, 3] + x
        self.new_extrinsic[1, 3] = self.extrinsic[1, 3] + y
        self.new_extrinsic[2, 3] = self.extrinsic[2, 3] + z

        # point_on_img, reflectance = proj_pcd2cam.get_pointcloud_on_image(self.intrisic, self.new_extrinsic, self.pointcloud)
        point_on_img, reflectance = proj_mapped_pointcloud2cam.get_pointcloud_on_image(self.intrisic, self.new_extrinsic, self.pose, self.pc, self.IMG_W, self.IMG_H)
        
        u, v, z = point_on_img
        
        # adjusted_image = cv2.merge([self.image[:,:,0] * b, self.image[:,:,1] * g, self.image[:,:,2] * r])

        if self.show_type == 'z':
            adjusted_image = draw_circle(self.image, u, v, z)
        elif self.show_type == 'reflectance':
            adjusted_image = draw_circle(self.image, u, v, reflectance)
        # Convert to PIL format and update the label
        pil_image = Image.fromarray(cv2.cvtColor(adjusted_image.astype('uint8'), cv2.COLOR_BGR2RGB))
        resize_pil_image = pil_image.resize((pil_image.size[0]//2, pil_image.size[1]//2))
        if image_save_path != '':
            resize_pil_image.save(image_save_path)
        tk_image = ImageTk.PhotoImage(resize_pil_image)
        self.image_label.configure(image=tk_image)
        self.image_label.image = tk_image

def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    i = int(h * 6.)  # 假设 H 已经是 [0, 1) 区间
    f = (h * 6.) - i
    p = v * (1. - s)
    q = v * (1. - s * f)
    t = v * (1. - s * (1. - f))
    i = i % 6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q
    
def spectrum_color(value):
    hue = (120 + value * 240) / 360
    saturation = 1  # 饱和度设为最大，即纯色
    value = 1  # 明度设为最大

    return hsv_to_rgb(hue, saturation, value)

def draw_circle(image, u, v, z):
    adjusted_image = image.copy()
    max_in_z = np.max(z)
    z = z/max_in_z

    width = image.shape[1]
    for i in range(len(u)):
        # color = (255 * z[i], 0, 0)
        color = spectrum_color(z[i])
        # color = spectrum_color(u[i]/width)
        color_255 = (color[0]*255, color[1]*255, color[2]*255)
        cv2.circle(adjusted_image, (int(u[i]), int(v[i])), radius=5, color=color_255, thickness=-1)
    return adjusted_image

# Main function to create the GUI
def main():

    # img_dir = 'ros_data/image'
    # pointcloud_dir = 'ros_data/pointcloud/'

    ros_bag_name = 'in_1221_2023-12-21-16-06-08'
    # ros_bag_name = 'in_lsy_2023-12-20-14-39-47'
    img_dir = f'correspond_data/after_mapping/{ros_bag_name}/image'
    pose_dir = f'correspond_data/after_mapping/{ros_bag_name}/pointcloud'
    # calib_data_path = 'ros_data/20231218_132035_autoware_lidar_camera_calibration.yaml'
    calib_data_path = 'self_data/avpslam/calibration/top_lidar_backright_cam/20231221_211833_autoware_lidar_camera_calibration.yaml'
    pcd_path = 'self_data/after_mapping_pointcloud/in_1221_2023-12-21-16-06-08/global_fil_ascii.pcd'

    root = tk.Tk()
    
    app = ExtrinsicAdjuster(root, img_dir, pose_dir, calib_data_path, pcd_path)
    root.mainloop()

if __name__ == '__main__':
    main()