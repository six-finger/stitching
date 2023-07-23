import cv2
import numpy as np
import os

if __name__ == '__main__':
    origin_path = ''
    file_name = 'DSC00325.JPG'
    img = cv2.imread(os.path.join(origin_path, file_name))
    h, w, _ = img.shape
    cv2.circle(img, (w//2, h//2), radius=5, color=(0,0,255), thickness=-1)
    
    cv2.imwrite(os.path.join(origin_path, 'ctr_'+file_name), img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    