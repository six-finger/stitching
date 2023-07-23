import cv2
import numpy as np
import math
import sys
import transforms3d as tfs

# from ortho_corr import chess_corners
from op_single.utils.coordinate_trans import *

def inv(M):
    return np.linalg.inv(M)
def tr(M):
    return np.transpose(M)
def concat(t, dim=0):
    return np.concatenate(t, dim)

def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()
def save(name, img):
    cv2.imwrite(name, img)

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
def get_trans():
    return tr(np.array([[0,0,0]]))
def get_intrinsic(fx, fy, w, h):
    return np.array([[fx, 0, w/2], [0, fy, h/2], [0, 0, 1]])
def get_homography(K, M_, M_gt):
    A = K @ M_
    B = K @ M_gt

    b1 = B[0:1, 0:4]
    b2 = B[1:2, 0:4]
    b3 = B[2:3, 0:4]
    h1 = inv(A @ tr(A)) @ A @ tr(b1)
    h2 = inv(A @ tr(A)) @ A @ tr(b2)
    h3 = inv(A @ tr(A)) @ A @ tr(b3)
    H = tr(concat((h1, h2, h3),1))
    return H

def transform_point(vecT, Mat):
    vecT = Mat @ tr(vecT)
    vecT = tr(vecT / vecT[2])
    return vecT
def transform_image(img, H):
    h, w, _ = img.shape
    xmin,xmax,ymin,ymax = get_img_corner(w, h, H)
    # 图像比例缩小
        # scale = 1000 / max(abs(min(lu-ld,ru-rd)), abs(max(lu-ld,ru-rd))) 
        # H = np.array([[scale,0,0], [0,scale,0], [0,0,1]]) @ H
        # lu,ld,ru,rd = image_corner(w, h, H)
    # 图像平移d
    H = np.array([[1,0,-xmin], [0,1,-ymin], [0,0,1]]) @ H
    xmin,xmax,ymin,ymax = get_img_corner(w, h, H)

    warped = cv2.warpPerspective(img, H, (int(xmax-xmin),int(ymax-ymin)))

    # save('/mnt/wangyue/project/ortho-rect/ortho_warp_.png', warped)
    # exit()
    return warped, H

def warp_pipeline(img, deg_vec, fx, fy):
    rad_vec = [deg2rad(i) for i in deg_vec]
    h, w, _ = img.shape        
    K = get_intrinsic(fx, fy, w, h)
    
    # 方法1
    # M_ = concat((get_rotation(rvec), get_trans()), 1)       # R_ = get_rotation(rvec)     
    # M_gt = concat((np.eye(3), np.zeros((3,1))), 1)          # R_gt = np.eye(3)
    # H = get_homography(K, M_, M_gt)
    
    #方法2    
    R1 = tfs.euler.euler2mat(rad_vec[0], rad_vec[1], rad_vec[2])
    H = K @ R1 @ inv(K)

    image_warped, H = transform_image(img, H)

    return image_warped, H

    

if __name__ == '__main__':
    # fx, fy = 3745, 3745         # 焦距
    fx, fy = 9542, 9542         # 焦距
    # rvec = [0, 30, 0]
    rvec = [2.11322192,1.518741388,-60.45357076]
    # img_path = r'/mnt/wangyue/project/ortho-rect/img/others/-30.jpg'
    img_path = r'/mnt/wangyue/project/ortho-rect/img/origin/DSC00038.JPG'

    image = cv2.imread(img_path)
    warped, H = warp_pipeline(image, rvec, fx, fy)

    new_name = img_path[0:img_path.rfind('.')]+'_warped2.jpg'
    print(new_name)
    # show('out', warped)
    save(new_name, warped)
    print(H)    