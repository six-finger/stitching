import cv2
import numpy as np
import glob


def chess_corners(img, chess_w, chess_h):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (chess_w, chess_h))
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners3 = np.array([])
    if ret == True:
        # corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        corners3 = np.array([corners[1],corners[chess_w-2],corners[chess_w*chess_h-chess_w+1],corners[chess_w*chess_h-2]])
        cv2.drawChessboardCorners(img, (2,2), corners3, ret)
    points = np.reshape(corners3,(4,2))
    # print('四个点的坐标是:\n', points)
    return points

def warped_image(img, points):
    dst = np.array([[500, 500], 
                    [700, 500],
                    [500, 700],
                    [700, 700]], dtype = "float32")
    M = cv2.getPerspectiveTransform(points, dst)
    print('变换矩阵是', M)
    warped = cv2.warpPerspective(img, M, (1000, 1000))
    return warped

def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()
def save(name, img):
    cv2.imwrite(name, img)

if __name__ == '__main__':
    file_name = '.\\102.jpg'
    chess_w = 8     # 棋盘格模板长边和短边规格（角点个数）
    chess_h = 6
    img = cv2.imread(file_name)

    corners = chess_corners(img, chess_w, chess_h)
    warped = warped_image(img, corners)
    new_name = file_name[0:file_name.rfind('.')]+'_corner.jpg'
    save(new_name, warped)