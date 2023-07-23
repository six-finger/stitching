# https://github.com/OpenStitching/stitching

from stitching import my_Stitcher, my_AffineStitcher
# from .stitching import my_image_handler
from op_single.ortho_warp import *
from op_single.utils.csv_tool import *

from pathlib import Path
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import os

global img_pool
img_pool = []
img_data = {
    'name': '',       
    'gps': [0,0,0],
    'rpy': [0,0,0],  
    'img': None,     
    'warped_img': None,
    'mask': None,
    'H': None
}
pos_file_path = r'F:\UCAS\projects\ortho_projects\stitching\data\1POS_RPY.csv'

def plot_image(img, figsize_in_inches=(5,5)):
    fig, ax = plt.subplots(figsize=figsize_in_inches)
    ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()
def plot_images(imgs, figsize_in_inches=(5,5)):
    fig, axs = plt.subplots(1, len(imgs), figsize=figsize_in_inches)
    for col, img in enumerate(imgs):
        axs[col].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()
    
def get_image_paths(img_dir, img_set):
    return [str(path.relative_to('.')) for path in Path(f'{img_dir}').rglob(f'{img_set}*')]


def get_img_corner(w, h, M):
    a = np.array([[0,0,1]],dtype=np.float32)
    b = np.array([[w,0,1]],dtype=np.float32)
    c = np.array([[0,h,1]],dtype=np.float32)
    d = np.array([[w,h,1]],dtype=np.float32)
    a_,b_,c_,d_ = transform_point(a,M), transform_point(b,M), transform_point(c,M), transform_point(d,M)
    A = concat((a_,b_,c_,d_))
    l0 = A[0:4, 0:1]
    l1 = A[0:4, 1:2]
    return l0.min(), l0.max(), l1.min(), l1.max()
def transform_point(vecT, Mat):
    vecT = Mat @ tr(vecT)
    vecT = tr(vecT / vecT[2])
    return vecT
def inv(M):
    return np.linalg.inv(M)
def tr(M):
    return np.transpose(M)
def concat(t, dim=0):
    return np.concatenate(t, dim)


def get_homography(img_name, image, fx, fy):
    csv_data = read_csv_file(pos_file_path)
    img_name = os.path.basename(img_name)
    row = get_specific_img(csv_data, img_name)
    gps = [float(num) for num in row[1:4]]
    rpy = [deg2rad(float(num)) for num in row[4:7]]
    warped_img, H = warp_pipeline(image, rpy, fx, fy)
    return H, warped_img, gps, rpy

def get_mask(image, H, warped_img):
    h, w, _ = image.shape
    lu = transform_point([0,0,1], H)[:2]
    ru = transform_point([w,0,1], H)[:2]
    rd = transform_point([w,h,1], H)[:2]
    ld = transform_point([0,h,1], H)[:2]

    mask = np.zeros_like(warped_img)     # 创建一个与原始图像相同大小的mask
    points = np.array([[lu,ru,rd,ld]], dtype=np.int32)   # 在mask上绘制bbox
    cv.fillPoly(mask, points, (255, 255, 255))
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    return mask

def push_img(img_name):
    img_bean = img_data.copy()
    img = cv.imread(img_name)
    img = cv.resize(img, None, None, 0.5, 0.5)
    h, w, _ = img.shape
    fx, fy = max(h, w), max(h, w)
    H, warped_img, gps, rpy = get_homography(img_name, img, fx, fy)
    mask = get_mask(img, H, warped_img)

    img_bean['name'] = img_name
    img_bean['gps'] = gps
    img_bean['rpy'] = rpy
    img_bean['img'] = img
    img_bean['warped_img'] = warped_img
    img_bean['mask'] = mask
    img_bean['H'] = H
    img_pool.append(img_bean)

def get_mask_by_name(img_name):
    for img_bean in img_pool:
        if img_bean['name'] == img_name:
            return img_bean['mask']
    return None

def get_img_by_name(img_name):
    for img_bean in img_pool:
        if img_bean['name'] == img_name:
            return img_bean['img']
    return None

def show_img(img, delay):
    img = cv.resize(img, (800,800))
    cv.imshow('wh', img)
    key = cv.waitKey(delay)
    if key == ord('q'):
        cv.destroyAllWindows()

if __name__ == '__main__':
    origin_img_dir = 'imgs'
    warped_img_dir = 'img_dir'
    img_set = 'DSC00'

    stitcher = my_Stitcher(crop=False, 
                            final_megapix=-1,
                            nfeatures=500,
                            estimator="affine",
                            matcher_type="affine",
                            adjuster="affine",
                            warper_type="affine",
                            wave_correct_kind="no",
                            compensator="no")

    # cv.ocl.setUseOpenCL(False)
    cv.ocl.setUseOpenCL(True)

    weihai_imgs = get_image_paths(origin_img_dir, img_set)
    try:
        for i, img_name in enumerate(weihai_imgs):
            # cv.ocl.finish()
            push_img(img_name)
            if i == 0:
                pass
            elif i == 1:
                img_name_lst = [i['name'] for i in img_pool[:2]]
                img_lst = [i['img'] for i in img_pool[:2]]
                mask_lst = [get_mask_by_name(i) for i in img_name_lst]
                panorama, pano_mask = stitcher.mask_stitch(img_name_lst, img_lst, mask_lst)
                show_img(panorama, 3)
            else:
                print([img_name])
                panorama, pano_mask = stitcher.mask_stitch(['  ',img_name], [panorama, get_img_by_name(img_name)], 
                                                           [pano_mask, get_mask_by_name(img_name)])
                                                        # incre_img=panorama, incre_mask=pano_mask)
                if i==len(weihai_imgs)-1:
                    show_img(panorama, 5000)
                else:
                    show_img(panorama, 3)
    except Exception as e :
        print(e)
    finally:
        cv2.imwrite(r'res_mask.jpg', panorama)
    