import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import glob
from tqdm import tqdm

def process_one_frame(number):
    with open(f'./testing/calib/{number}.txt','r') as f:
        calib = f.readlines()
    # 相机内参 camera2 
    # | fx 0 u0 |
    # | 0 fy v0 |
    # | 0  0  1 |
    # example: 000007.txt
    # 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 4.485728000000e+01 
    # 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.163791000000e-01 
    # 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.745884000000e-03
    P2 = np.array([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3,4)

    # 相机旋转矩阵
    # | r11  r12  r13 |
    # | r21  r22  r23 |
    # | r31  r32  r33 |
    # example: 000007.txt
    # ——————————————————————————————————————————————————————————————————
    # 9.999239000000e-01 9.837760000000e-03 -7.445048000000e-03 
    # -9.869795000000e-03 9.999421000000e-01 -4.278459000000e-03 
    # 7.402527000000e-03 4.351614000000e-03 9.999631000000e-01
    # ——————————————————————————————————————————————————————————————————
    R0_rect = np.array([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3,3)

    # 矩阵转为如下形式, 增加维度方便计算
    # ——————————————————————————————————————————————————————————————————
    #  9.999239000000e-01  9.837760000000e-03 -7.445048000000e-03 0
    # -9.869795000000e-03  9.999421000000e-01 -4.278459000000e-03 0
    #  7.402527000000e-03  4.351614000000e-03  9.999631000000e-01 0
    #                   0                   0                   0 1
    # ——————————————————————————————————————————————————————————————————
    R0_rect = np.insert(R0_rect,3,values=[0,0,0],axis=0)
    R0_rect = np.insert(R0_rect,3,values=[0,0,0,1],axis=1)

    # 激光雷达到相机坐标变换矩阵
    # | R  T |
    # | 0  1 |
    # example: 000007.txt
    # 7.533745000000e-03 -9.999714000000e-01 -6.166020000000e-04 -4.069766000000e-03 
    # 1.480249000000e-02 7.280733000000e-04 -9.998902000000e-01 -7.631618000000e-02 
    # 9.998621000000e-01 7.523790000000e-03 1.480755000000e-02 -2.717806000000e-01
    Tr_velo_to_cam = np.array([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3,4)

    # 7.533745000000e-03 -9.999714000000e-01 -6.166020000000e-04 -4.069766000000e-03 
    # 1.480249000000e-02  7.280733000000e-04 -9.998902000000e-01 -7.631618000000e-02 
    # 9.998621000000e-01  7.523790000000e-03  1.480755000000e-02 -2.717806000000e-01
    #                  0                   0                   0                   1
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam,3,values=[0,0,0,1],axis=0)

    # read point cloud file
    point_cloud_file = f'./data_object_velodyne/testing/velodyne/{number}.bin'
    scan = np.fromfile(point_cloud_file, dtype=np.float32).reshape((-1,4))

    # lidar xyz (front, left, up)
    points = scan[:, 0:3]

    # reflectance
    reflectance = scan[:, 3].T

    # 补充一个维度，便于矩阵计算
    velo = np.insert(points,3,1,axis=1).T

    # 删除距离为负的点云
    reflectance = np.delete(reflectance, np.where(velo[0,:]<0))
    velo = np.delete(velo,np.where(velo[0,:]<0),axis=1)

    # 点云 [x y z] 转 [u v]
    #
    #           [fx   0   u0   ?]     [r11  r12  r13  0]     [          ]
    #  cam =    [0    fy  v0   ?]  *  [r21  r22  r23  0]  *  [  R     t ]  *  velo(4, n)
    #           [0    0    1   ?]     [r31  r32  r33  0]     [          ]
    #                                 [0    0    0    1]     [  0     1 ]
    cam = P2.dot(R0_rect.dot(Tr_velo_to_cam.dot(velo)))

    # 删除像方坐标z为负值的点
    reflectance = np.delete(reflectance, np.where(cam[2,:]<0))
    cam = np.delete(cam,np.where(cam[2,:]<0),axis=1)

    # get u,v,z
    cam[:2] /= cam[2,:]

    # plt init
    img_file = f'./data_object_image_2/testing/image_2/{number}.png'
    fig, axes = plt.subplots(3, 1)
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

    # filter point out of canvas 删除相机取景框以外的点云
    u,v,z = cam
    u_out = np.logical_or(u<0, u>IMG_W)
    v_out = np.logical_or(v<0, v>IMG_H)
    outlier = np.logical_or(u_out, v_out)

    reflectance = np.delete(reflectance,np.where(outlier))
    cam = np.delete(cam,np.where(outlier),axis=1)

    # 根据 u, v 将点云画到图像上 (s可调整点云像素大小)
    u,v,z = cam
    axes[1].scatter([u],[v],c=[z],cmap='rainbow_r',alpha=0.5,s=1)
    axes[2].scatter([u],[v],c=[reflectance],cmap='rainbow_r',alpha=0.5,s=1)

    os.makedirs('./data_object_image_2/testing/projection', exist_ok=True)
    plt.savefig(f'./data_object_image_2/testing/projection/{number}.png',dpi=300,bbox_inches='tight')

    # plt.show()

def main():
    img_dir = "data_object_image_2/testing/image_2"
    img_nums = []
    for file_path in glob.glob(os.path.join(img_dir, '*.png')):  # 替换 '*.png' 为你想要匹配的图片格式，比如 '*.jpg'
        img_nums.append(os.path.splitext(os.path.basename(file_path))[0])
    
    for num in tqdm(img_nums):
        process_one_frame(num)

if __name__ == '__main__':
    main()