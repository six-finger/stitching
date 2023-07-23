import cv2 as cv
import matplotlib.pyplot as plt


def detect_sift(img):
    sift = cv.SIFT_create() # SIFT特征提取对象
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # 转为灰度图
    kp = sift.detect(gray, None) # 关键点位置
    kp = sorted(kp, key=lambda x: x.response, reverse=True)[:10000]
    kp, des = sift.compute(gray, kp) # des为特征向量
    print(des.shape) # 特征向量为128维
    return kp, des

print(0)
img1 = cv.imread('DSC00120_warped.jpg')
print(1)
img2 = cv.imread('DSC00122_warped.jpg')
print(2)
kp1, des1 = detect_sift(img1)
print(3)
kp2, des2 = detect_sift(img2)
print(4)

bf = cv.BFMatcher(crossCheck=True) # 匹配对象
matches = bf.match(des1, des2) # 进行两个特征矩阵的匹配
res = cv.drawMatches(img1, kp1, img2, kp2, matches, None) # 绘制匹配结果
import op_single.ortho_warp as ortho_warp
ortho_warp.save('match.jpg', res)

cv.imshow('res', res)
cv.waitKey(0)
cv.destroyAllWindows()