from stitching import Stitcher, AffineStitcher
from pathlib import Path
from matplotlib import pyplot as plt
import cv2 as cv


def get_image_paths(img_dir, img_set):
    return [str(path.relative_to('.')) for path in Path(f'{img_dir}').rglob(f'{img_set}*')]


if __name__ == '__main__':
    origin_img_dir = 'imgs'
    warped_img_dir = 'img_dir'
    img_set = 'DSC00'

    stitcher = Stitcher(crop=False, 
                        final_megapix=-1,
                        nfeatures=2000,
                        estimator="affine",
                        matcher_type="affine",
                        adjuster="affine",
                        warper_type="affine",
                        wave_correct_kind="no",
                        compensator="gain_blocks")
    
    cv.ocl.setUseOpenCL(True)

    weihai_imgs = get_image_paths(origin_img_dir, img_set)[:15]
    print(weihai_imgs)
    panorama = stitcher.stitch_once(weihai_imgs)
    cv.imwrite(r'res_once.jpg', panorama)
