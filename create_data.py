import sys
import os
import glob
import shutil

# create_data.py
# 功能:
# 1.重命名图像
# 2.将点云文件和图像文件根据时间戳一一对应, 拷贝到当前文件夹下的correspond_data文件夹下
def main():
    sub_dir_name = 'in_1221_2023-12-21-16-06-08'
    images_dir = os.path.join('self_data/avpslam/tongji_data/image/',sub_dir_name)
    pointclouds_dir = os.path.join('self_data/avpslam/tongji_data/pointcloud',sub_dir_name)
    dst_dir = os.path.join('./correspond_data/',sub_dir_name)
    # 图像重命名
    rename_images(images_dir)

    produce_one_to_one_data(images_dir, pointclouds_dir, dst_dir)
    
    print('finished!')

def rename_images(folder_path):
    files = []

    for file_path in glob.glob(os.path.join(folder_path, '*.jpg')):
        basename =  os.path.splitext(os.path.basename(file_path))[0]
        if '.' in basename:
            files.append(basename)

    sorted(files)

    for filename in files:
        new_filename = str(int(float(filename) * 1000000))

        original_path = os.path.join(folder_path, filename + '.jpg')
        new_path = os.path.join(folder_path, new_filename + '.jpg')

        os.rename(original_path, new_path)
        print(f"Renamed {filename} to {new_filename}")


def produce_one_to_one_data(images_dir, pointclouds_dir, dst_dir):
    images_name = []
    for file_path in glob.glob(os.path.join(images_dir, '*.jpg')):
        images_name.append(os.path.splitext(os.path.basename(file_path))[0])
    
    images_name.sort(key=lambda x: int(x))
    
    pointclouds_name = []
    for file_path in glob.glob(os.path.join(pointclouds_dir, '*.pcd')):
        pointclouds_name.append(os.path.splitext(os.path.basename(file_path))[0])

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
        img_path = os.path.join(images_dir, img_file + '.jpg')
        pc_path = os.path.join(pointclouds_dir, correspoind_pt + '.pcd')
        
        dst_pc_path = os.path.join(dst_pc_dir, correspoind_pt + '.pcd')
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
    for file2 in file2_list:
        file2_time = float(file2)
        cur = abs(file2_time - file1_time)
        if cur > last:
            break
        if cur < min_time:
            min_time = abs(file2_time - file1_time)
            result = file2
        last = cur
    
    return result

if __name__ == '__main__':
    main()