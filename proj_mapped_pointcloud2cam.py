import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import glob
from tqdm import tqdm
import yaml
from PIL import Image


from scipy.spatial.transform import Rotation as R
from scipy.ndimage import minimum_filter
from scipy.ndimage import generic_filter

import time
import shutil
import cv2

# 此程序较之前程序，多了一步把全局点云转到局部坐标系的步骤，需要得到最接近当前图片时间戳的pose文件

def load_pcd_data(file_path):
    pts = []
    with open(file_path,'r') as f:
        data = f.readlines()
    line = data[9]
    line = line.strip('\n')
    i = line.split(' ')
    pts_num = eval(i[-1])
    for line in tqdm(data[11:]):
        line = line.strip('\n')
        xyzi = line.split(' ')
        if len(xyzi) == 4:
        # print(line)
            if xyzi[0] != 'nan' and xyzi[1] != 'nan' and xyzi[2] != 'nan' and xyzi[3] != 'nan':
                x, y, z = [eval(i) for i in xyzi[0:3]]
                intensity = str(int(xyzi[3])/255.0)
                pts.append([x, y, z, intensity])
        elif len(xyzi) == 5:
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

    print(f"origin points: {pts_num}")
    print(f"valid points: {len(pts)}")
    print(f"invalid points: {pts_num - len(pts)}")
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

def height_filter(pointcloud, lower, upper, reflectance):
    new_pointcloud = pointcloud.copy()
    new_pointcloud = new_pointcloud.T
    reflectance = np.delete(reflectance, np.where(new_pointcloud[:,2]>upper), axis=0)
    new_pointcloud = np.delete(new_pointcloud,np.where(new_pointcloud[:,2]>upper), axis=0)
    reflectance = np.delete(reflectance, np.where(new_pointcloud[:,2]<lower), axis=0)
    new_pointcloud = np.delete(new_pointcloud,np.where(new_pointcloud[:,2]<lower), axis=0)

    return new_pointcloud.T, reflectance


def remove_background_points(points, u_threshold=20, v_threshold=30, z_diff_threshold=2):

    i = 0
    # indexs = []
    while i < len(points):
        u, v, z, _ = points[i]
        j = i+1
        print(f'i: {i} total: {len(points)}')
        while j < len(points):
            u_prime, v_prime, z_prime, _ = points[j]
            # u_distance = u_prime - u
            # v_distance = v_prime - v
            pixel_distance = (u_prime - u)**2 + (v_prime - v)**2
            spatial_distance = abs(z_prime - z)

            index = spatial_distance / ((u_prime - u)**2 + (v_prime - v)**2)

            if (pixel_distance < u_threshold**2 and spatial_distance > z_diff_threshold) or index > 0.001:
                # Remove the point with the larger z value
                print(index)

                if z > z_prime:
                    points = np.delete(points, i, axis=0)
                    i = i-1
                    break
                else:
                    points = np.delete(points, j, axis=0)
                    j = j-1
            j = j+1
        i = i+1

    return points

def apply_mask(matrix1, matrix2):
    # 创建一个与 matrix2 相同形状的零矩阵
    masked_matrix = np.zeros_like(matrix2)
    
    # 使用掩膜矩阵中非零元素的索引来更新 masked_matrix
    masked_matrix[matrix1 != 0] = matrix2[matrix1 != 0]
    
    return masked_matrix

def extract_nonzero_pixels(image):
    # 获取图像中所有非零元素的索引
    nonzero_indices = np.nonzero(np.any(image != 0, axis=2))
    
    # 提取非零元素的坐标和像素值
    nonzero_pixels = np.vstack((nonzero_indices[1], nonzero_indices[0], image[nonzero_indices][:,0], image[nonzero_indices][:,1])).T
    
    return nonzero_pixels

def min_filter_exclude_zeros(image, filter_size = 5):
    # 定义一个函数来处理每个窗口
    def min_nonzero_window(window):
        nonzero_values = window[window != 0]
        mid = len(window)//2
        if len(nonzero_values) > 0:
            if window[mid] != np.min(nonzero_values):
                return 0
            else:
                return np.min(nonzero_values)
        else:
            return 0

    # 使用 generic_filter 进行窗口滤波
    filtered_image = generic_filter(image, min_nonzero_window, size=filter_size)

    return filtered_image
    
def conv_remove_background_points(points, filter_size=5, z_diff_threshold=2):

    # 计算矩阵的大小
    width = 1920
    height = 1200

    # 创建一个空的 NumPy 矩阵
    matrix = np.zeros((height, width, 2))

    # 将像素值填充到矩阵中
    for i in range(points.shape[0]):
        x, y, z, reflectance= points[i]
        matrix[int(y), int(x)][0] = z
        matrix[int(y), int(x)][1] = reflectance

    # 对z值进行窗口滤波
    image = matrix[:,:,0]
    
    filtered_z_matrix = min_filter_exclude_zeros(image, filter_size=filter_size)

    reflectance_matrix = matrix[:,:,1]

    filtered_reflectance_matrix = apply_mask(filtered_z_matrix, reflectance_matrix)

    filter_matrix = np.dstack((filtered_z_matrix, filtered_reflectance_matrix))
    
    points = extract_nonzero_pixels(filter_matrix)

    return points

def get_pointcloud_on_image(intrinsic, extrinsic, pose_matrix, pointcloud, IMG_W, IMG_H):
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
    inverse_pose_matrix = np.linalg.inv(pose_matrix)

    # top lidar to vehicle
    lidar_pose = [0.00733038, -0.00349064, 1.27939e-05, 0.059, 0.01, 0.86]
    theta_x, theta_y, theta_z, dx, dy, dz = lidar_pose[0:6]
    lidar_pose_transform_matrix = pose_to_transform_3d(dx, dy, dz, theta_x, theta_y, theta_z)
    inverse_lidar_pose_transform_matrix = np.linalg.inv(lidar_pose_transform_matrix)

    transformered_pc = inverse_lidar_pose_transform_matrix.dot(inverse_pose_matrix.dot(lidar))

    # 过滤掉非地面的点云数据
    transformered_pc, reflectance = height_filter(transformered_pc, -1.2, -0.85, reflectance)

    cam = intrinsic.dot(extrinsic.dot(transformered_pc))

    # cam = intrinsic.dot(extrinsic.dot(pose_matrix.dot(lidar)))
    # cam = intrinsic.dot(extrinsic.dot(lidar))

    # 删除像方坐标z为负值的点
    reflectance = np.delete(reflectance, np.where(cam[2,:]<0))
    cam = np.delete(cam,np.where(cam[2,:]<0),axis=1)

    # get u,v,z
    cam[:2] /= cam[2,:]
    # 删除取景框外的点
    u,v,z = cam
    u_out = np.logical_or(u<0, u>IMG_W)
    v_out = np.logical_or(v<0, v>IMG_H)
    outlier = np.logical_or(u_out, v_out)

    reflectance = np.delete(reflectance, np.where(outlier))
    cam = np.delete(cam,np.where(outlier),axis=1)

    # 删除背景点云（透视的点云）
    reflectance = np.reshape(reflectance, (1,reflectance.shape[0]))
    points = np.concatenate((cam.T, reflectance.T), axis=1)
    # 方法1
    points = conv_remove_background_points(points, filter_size=5, z_diff_threshold=0.15)
    # 方法2
    # points = remove_background_points(points, u_threshold=20, v_threshold=20, z_diff_threshold=0.15)
    cam = points[:,:3].T
    reflectance = points[:,3].T

    return cam, reflectance

def plt_init(img_file):
    fig, axes = plt.subplots(1, 3)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    img = mpimg.imread(img_file)
    IMG_H,IMG_W,_ = img.shape

    axes[0].imshow(img)
    axes[0].set_title('Image', fontsize=6)

    axes[1].imshow(img)
    axes[1].set_title('Depth Mix', fontsize=6)

    axes[2].imshow(img)
    axes[2].set_title('Reflectance Mix', fontsize=6)

    plt.tight_layout()
    axes[0].axis('off')
    axes[1].axis('off')
    axes[2].axis('off')
    return IMG_H, IMG_W, axes

def pose_to_transform_3d(dx, dy, dz, theta_x, theta_y, theta_z):
        # Translation matrix
        translation_matrix = np.eye(4)
        translation_matrix[0, 3] = dx
        translation_matrix[1, 3] = dy
        translation_matrix[2, 3] = dz

        # Rotation matrix
        rotation_matrix = R.from_euler('xyz', [theta_x, theta_y, theta_z], degrees=False).as_matrix()
        
        # Combine translation and rotation
        transform_matrix = np.dot(translation_matrix, np.vstack([np.hstack([rotation_matrix, np.zeros((3, 1))]), [0, 0, 0, 1]]))
        
        return transform_matrix

def get_pose_matrix(pose_file):
    with open(pose_file, 'r') as f:
        line = f.readline()
        data = line.split(' ')
        theta_x, theta_y, theta_z, dx, dy, dz = data[0:6]

    transform_matrix = pose_to_transform_3d(dx, dy, dz, theta_x, theta_y, theta_z)

    return transform_matrix

def window_filter(image_path, window_size=5):
    # 读取图像
    image = cv2.imread(image_path)

    # 获取图像尺寸
    height, width, channels = image.shape

    # 创建一个和原始图像大小相同的空白图像
    filtered_image = np.zeros_like(image)

    # 遍历图像像素
    for y in range(height):
        for x in range(width):
            # 获取滤波窗口范围
            x_start = max(0, x - window_size // 2)
            x_end = min(width, x + window_size // 2 + 1)
            y_start = max(0, y - window_size // 2)
            y_end = min(height, y + window_size // 2 + 1)
            
            # 获取窗口内像素值
            window_pixels = image[y_start:y_end, x_start:x_end]
            
            # 计算窗口内最小值
            min_value = np.min(window_pixels)
            
            # 将整个窗口设置为最小值
            filtered_image[y, x] = min_value

    return filtered_image

def get_closest_point(u, v, z):
    # 生成示例数据
    image = np.full((1200,1920), 255, np.uint8)

    for i in range(len(u)):
        image[int(v[i])][int(u[i])] = z[i]

    window_size = 10

    # 构建最小值滤波核
    kernel = np.ones((window_size, window_size), np.uint8)

    # 对图像进行最小值滤波
    filtered_image = cv2.erode(image, kernel)

    # 显示原始图像和滤波后的图像
    cv2.imshow('Original Image', image)
    cv2.imshow('Filtered Image', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
    

def process_one_frame(img_file, pose_file, cam_lidar_calib_file, scan, create_rgbi_data=False):
    # 读取标定得到的内外参
    # cam_lidar_calib_file='ros_data/20231218_132035_autoware_lidar_camera_calibration.yaml'
    # cam_lidar_calib_file = 'self_data/avpslam/calibration/top_lidar_backright_cam/20231221_211833_autoware_lidar_camera_calibration.yaml'
    intrinsic, extrinsic = get_calib_param(cam_lidar_calib_file)
    # plt init
    img_name = os.path.splitext(os.path.basename(img_file))[0]
    img = mpimg.imread(img_file)
    IMG_H,IMG_W,_ = img.shape
    # IMG_H, IMG_W, axes = plt_init(img_file)

    pose_matrix = get_pose_matrix(pose_file)
    cam, reflectance = get_pointcloud_on_image(intrinsic, extrinsic, pose_matrix, scan, IMG_W, IMG_H)


    # filter point out of canvas 删除相机取景框以外的点云
    # u,v,z = cam
    # u_out = np.logical_or(u<0, u>IMG_W)
    # v_out = np.logical_or(v<0, v>IMG_H)
    # outlier = np.logical_or(u_out, v_out)

    # reflectance = np.delete(reflectance, np.where(outlier))
    # cam = np.delete(cam,np.where(outlier),axis=1)

    # 根据 u, v 将点云画到图像上 (s可调整点云像素大小)
    u,v,z = cam
    
    # depth_map = get_closest_point(u, v, z)
    # axes[1].scatter([u],[v],c=[z],cmap='rainbow_r',alpha=0.5,s=5)
    # axes[2].scatter([u],[v],c=[reflectance],cmap='rainbow_r',alpha=0.5,s=5)

    # projection_save_dir = 'ros_data/projection/'
    # os.makedirs(projection_save_dir, exist_ok=True)
    # # plt.savefig(f'./data_object_image_2/testing/projection/{number}.png',dpi=300,bbox_inches='tight')
    # plt.savefig(os.path.join(projection_save_dir, img_name),dpi=300,bbox_inches='tight')

    # plt.show()
    if create_rgbi_data == True:
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

def rename_files(folder_path):
    files = []

    for file_path in glob.glob(os.path.join(folder_path, '*.pose')):
        basename =  os.path.splitext(os.path.basename(file_path))[0]
        if int(basename)>1e17:
            files.append(basename)

    sorted(files)

    for filename in files:
        new_filename = str(int(filename) // 1000)

        original_path = os.path.join(folder_path, filename + '.pose')
        new_path = os.path.join(folder_path, new_filename + '.pose')

        os.rename(original_path, new_path)
        print(f"Renamed {filename} to {new_filename}")

def produce_one_to_one_data(images_dir, pointclouds_dir, dst_dir):
    images_name = []
    for file_path in glob.glob(os.path.join(images_dir, '*.jpg')):
        images_name.append(os.path.splitext(os.path.basename(file_path))[0])
    
    images_name.sort(key=lambda x: int(x))
    
    pointclouds_name = []
    # for file_path in glob.glob(os.path.join(pointclouds_dir, '*.pcd')):
    for file_path in glob.glob(os.path.join(pointclouds_dir, '*.pose')):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        pointclouds_name.append(file_name)

    pointclouds_name.sort(key=lambda x: int(x))

    
    dst_img_dir = os.path.join(dst_dir, 'image')
    dst_pc_dir = os.path.join(dst_dir, 'pointcloud')
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_pc_dir, exist_ok=True)

    # 图像帧率高于点云帧率时
    # for pc_file in pointclouds_name:
    #     correspoind_img = find_nearest(pc_file, images_name)
    #     pc_path = os.path.join(pointclouds_dir, pc_file + '.pcd')
    #     img_path = os.path.join(images_dir, correspoind_img + '.jpg')
        
    #     dst_pc_path = os.path.join(dst_pc_dir, pc_file + '.pcd')
    #     dst_img_path = os.path.join(dst_img_dir, correspoind_img + '.jpg')
    #     shutil.copy(img_path, dst_img_path)
    #     print(f"Copied {img_path} to {dst_img_path}")
    #     shutil.copy(pc_path, dst_pc_path)
    #     print(f"Copied {pc_path} to {dst_pc_path}")

    # 点云帧率高于图像帧率时
    for img_file in images_name:
        correspoind_pt = find_nearest(img_file, pointclouds_name)
        if correspoind_pt == 0:
            continue
        img_path = os.path.join(images_dir, img_file + '.jpg')
        # pc_path = os.path.join(pointclouds_dir, correspoind_pt + '.pcd')
        pc_path = os.path.join(pointclouds_dir, correspoind_pt + '.pose')
        
        # dst_pc_path = os.path.join(dst_pc_dir, correspoind_pt + '.pcd')
        dst_pc_path = os.path.join(dst_pc_dir, correspoind_pt + '.pose')
        dst_img_path = os.path.join(dst_img_dir, img_file + '.jpg')
        shutil.copy(img_path, dst_img_path)
        print(f"Copied {img_path} to {dst_img_path}")
        shutil.copy(pc_path, dst_pc_path)
        print(f"Copied {pc_path} to {dst_pc_path}")


# 在file2列表中找到时间最接近file1的文件
def find_nearest(file1, file2_list):
    min_time = sys.maxsize
    file1_time = float(file1)
    last = abs(float(file2_list[0]) - file1_time)
    if float(file2_list[0]) > file1_time:
        return 0
    for file2 in file2_list:
        file2_time = float(file2)
        cur = abs(file2_time - file1_time)
        if cur > last or file2_time > file1_time:
            break
        if cur < min_time:
            min_time = cur
            result = file2
        last = cur
    
    return result

def main():
    sub_dir_name = 'in_1221_2023-12-21-16-06-08'
    images_dir = os.path.join('self_data/avpslam/tongji_data/image/',sub_dir_name)
    pointclouds_dir = f'self_data/after_mapping_pointcloud/{sub_dir_name}/global_filtered'
    dst_dir = os.path.join('./correspond_data/after_mapping',sub_dir_name)

    # rename_files(pointclouds_dir)

    # 图像与点云文件按照文件名做匹配，输出到 dst_dir 中
    # produce_one_to_one_data(images_dir, pointclouds_dir, dst_dir)
    
    # print('finished!')

    pointclouds_file = 'self_data/after_mapping_pointcloud/in_1221_2023-12-21-16-06-08/global_fil_ascii.pcd'
    pc = load_pcd_data(pointclouds_file)

    imgs_dir = 'correspond_data/after_mapping/in_1221_2023-12-21-16-06-08/image'
    poses_dir = 'correspond_data/after_mapping/in_1221_2023-12-21-16-06-08/pointcloud'

    images_name = []
    for file_path in glob.glob(os.path.join(imgs_dir, '*.jpg')):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        images_name.append(file_name)
    
    images_name.sort(key=lambda x: int(x))
    
    poses_name = []
    # for file_path in glob.glob(os.path.join(pointclouds_dir, '*.pcd')):
    for file_path in glob.glob(os.path.join(poses_dir, '*.pose')):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        poses_name.append(file_name)

    poses_name.sort(key=lambda x: int(x))
    
    for i in range(len(images_name)):
        image = os.path.join(imgs_dir, images_name[i]+'.jpg')
        pose = os.path.join(poses_dir, poses_name[i]+'.pose')
        process_one_frame(image, pose, pc)

if __name__ == '__main__':
    main()