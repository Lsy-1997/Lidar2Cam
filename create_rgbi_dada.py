import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import glob
from tqdm import tqdm
import yaml
from PIL import Image

def window_max_filter(img_array):
    from scipy.ndimage import maximum_filter

    # 定义窗口大小
    size = 15  # 3x3 窗口

    # 应用最大值滤波
    filtered_img_array = maximum_filter(img_array, size=size)

    return filtered_img_array


def load_pcd_data(file_path):
    pts = []
    with open(file_path,'r') as f:
        data = f.readlines()
    line = data[9]
    line = line.strip('\n')
    i = line.split(' ')
    pts_num = eval(i[-1])
    for line in data[11:]:
        line = line.strip('\n')
        xyzi = line.split(' ')
        if len(xyzi) == 4:
        # print(line)
            if xyzi[0] != 'nan' and xyzi[1] != 'nan' and xyzi[2] != 'nan' and xyzi[3] != 'nan':
                x, y, z = [eval(i) for i in xyzi[0:3]]
                intensity = str(int(xyzi[3])/255.0)
                pts.append([x, y, z, intensity])
        elif len(xyzi) == 6:
            if xyzi[0] != 'nan' and xyzi[1] != 'nan' and xyzi[2] != 'nan' and xyzi[3] != 'nan' and xyzi[4] != 'nan' and xyzi[5] != 'nan':
                x, y, z = [eval(i) for i in xyzi[0:3]]
                intensity = str(int(xyzi[3])/255.0)
                # lidar_line_num = xyzi[4]
                pts.append([x, y, z, intensity])

    # print(f"origin points: {pts_num}")
    # print(f"valid points: {len(pts)}")
    # print(f"invalid points: {pts_num - len(pts)}")
    res = np.zeros((len(pts), len(pts[0])), dtype=np.float32)
    for i in range(len(pts)):
        res[i] = pts[i]
    return res

def get_calib_param(cam_lidar_calib_file):
    with open(cam_lidar_calib_file, 'r') as file:
        cam_lidar_calib_data = yaml.safe_load(file)
    camera_mat = cam_lidar_calib_data.get('CameraMat')
    intrinsic = np.array(camera_mat.get('data')).reshape(3,3)
    intrinsic = np.insert(intrinsic,3,values=[0,0,0],axis=1)

    extrinsic = cam_lidar_calib_data.get('CameraExtrinsicMat')
    extrinsic = np.array(extrinsic.get('data')).reshape(4,4)
    return intrinsic, extrinsic

def transform_from_autoware_to_normal(extrinsic):
    extrinsic_new = extrinsic.copy()
    extrinsic_new[:3,:3] = extrinsic[:3,:3].T
    x = extrinsic[0, 3]
    y = extrinsic[1, 3]
    z = extrinsic[2, 3]
    extrinsic_new[0, 3] = y
    extrinsic_new[1, 3] = z
    extrinsic_new[2, 3] = -x

    return extrinsic_new

def get_pointcloud_on_image(intrinsic, extrinsic, pointcloud):
    # Autoware标定矩阵变换，跟普通变换矩阵不同, 若为普通变换矩阵，请注释下面这行代码
    extrinsic = transform_from_autoware_to_normal(extrinsic)

    # lidar xyz (front, left, up)
    points = pointcloud[:, 0:3]

    # reflectance
    reflectance = pointcloud[:, 3].T

    # 补充一个维度，便于矩阵计算
    lidar = np.insert(points,3,1,axis=1).T

    # 删除距离为负的点云
    # reflectance = np.delete(reflectance, np.where(velo[0,:]<0))
    # velo = np.delete(velo,np.where(velo[0,:]<0),axis=1)

    cam = intrinsic.dot(extrinsic.dot(lidar))

    # 删除像方坐标z为负值的点
    reflectance = np.delete(reflectance, np.where(cam[2,:]<0))
    cam = np.delete(cam,np.where(cam[2,:]<0),axis=1)

    # get u,v,z
    cam[:2] /= cam[2,:]

    return cam, reflectance

def process_one_frame(image_file, pointcloud_file, cam_lidar_calib_file):
    # 读取标定得到的内外参
    intrinsic, extrinsic = get_calib_param(cam_lidar_calib_file)

    # 读取激光点云数据
    scan = load_pcd_data(pointcloud_file)

    cam, reflectance = get_pointcloud_on_image(intrinsic, extrinsic, scan)

    img_name = os.path.splitext(os.path.basename(image_file))[0]

    # filter point out of canvas 删除相机取景框以外的点云
    img = mpimg.imread(image_file)
    IMG_H,IMG_W,_ = img.shape

    u,v,z = cam
    u_out = np.logical_or(u<0, u>IMG_W)
    v_out = np.logical_or(v<0, v>IMG_H)
    outlier = np.logical_or(u_out, v_out)

    reflectance = np.delete(reflectance,np.where(outlier))
    cam = np.delete(cam,np.where(outlier),axis=1)

    # 根据 u, v 将点云画到图像上 (s可调整点云像素大小)
    u,v,z = cam
    reflectance_img = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    for i in range(len(u)):
        reflectance_img[int(v[i])][int(u[i])] = int(reflectance[i]*255)

    # 显示点云投影图
    # reflectance_img_test = Image.fromarray(reflectance_img)
    # reflectance_img_test.show()

    img_array = np.array(img)

    # 点云投影图上采样，采用窗口极大值滤波
    # reflectance_img = window_max_filter(reflectance_img)

    # 将强度通道添加到原始 RGB 图像上
    reflectance_img = np.expand_dims(reflectance_img, axis=2)
    rgbi_img = np.concatenate((img_array, reflectance_img), axis=2)


    # visualized_rgbi_img = rgbi_img.copy()
    # visualized_rgbi_img[:,:,3][visualized_rgbi_img[:,:,3]==0] = 255
    # visualized_rgbi_img = Image.fromarray(visualized_rgbi_img)
    # visualized_rgbi_img.show()

    # 将 NumPy 数组转换回 PIL 图像并保存
    rgbi_img = Image.fromarray(rgbi_img, 'RGBA')
    rgbi_save_dir = 'rgbi_dataset'
    os.makedirs(rgbi_save_dir, exist_ok=True)
    rgbi_img.save(os.path.join(rgbi_save_dir, img_name + '.png'))  # 保存为 PNG
    

def main():

    ros_bag_name = 'in_1221_2023-12-21-16-06-08'
    img_dir = f'correspond_data/{ros_bag_name}/selected_image'
    pointcloud_dir = f'correspond_data/{ros_bag_name}/selected_pointcloud'
    # calib_data_path = 'ros_data/20231218_132035_autoware_lidar_camera_calibration.yaml'
    calib_data_path = 'self_data/avpslam/calibration/top_lidar_backright_cam/20231221_211833_autoware_lidar_camera_calibration.yaml'

    images_file = []
    for file_path in glob.glob(os.path.join(img_dir, f'*.jpg')):
        images_file.append(os.path.splitext(os.path.basename(file_path))[0])

    pointcloud_file = []
    for file_path in glob.glob(os.path.join(pointcloud_dir, f'*.pcd')):
        pointcloud_file.append(os.path.splitext(os.path.basename(file_path))[0])

    images_file.sort(key=lambda x: int(x))
    pointcloud_file.sort(key=lambda x: int(x))

    for i in tqdm(range(len(images_file))):
        image_path = os.path.join(img_dir, images_file[i] + ".jpg")
        pointcloud_path = os.path.join(pointcloud_dir, pointcloud_file[i] + ".pcd")
        process_one_frame(image_path, pointcloud_path, calib_data_path)
    print("finished!")

if __name__ == '__main__':
    main()