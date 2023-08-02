from statistics import median

import cv2 as cv
import numpy as np


class Warper:
    """https://docs.opencv.org/4.x/da/db8/classcv_1_1detail_1_1RotationWarper.html"""

    WARP_TYPE_CHOICES = (
        "spherical",
        "plane",
        "affine",
        "cylindrical",
        "fisheye",
        "stereographic",
        "compressedPlaneA2B1",
        "compressedPlaneA1.5B1",
        "compressedPlanePortraitA2B1",
        "compressedPlanePortraitA1.5B1",
        "paniniA2B1",
        "paniniA1.5B1",
        "paniniPortraitA2B1",
        "paniniPortraitA1.5B1",
        "mercator",
        "transverseMercator",
    )

    DEFAULT_WARP_TYPE = "plane"

    def __init__(self, warper_type=DEFAULT_WARP_TYPE):
        self.warper_type = warper_type
        self.scale = None

    def set_scale(self, cameras):
        focals = [cam.focal for cam in cameras]
        self.scale = median(focals)

    def warp_images(self, imgs, cameras, aspect=1):
        for img, camera in zip(imgs, cameras):
            yield self.warp_image(img, camera, aspect)

    def warp_image(self, img, camera, aspect=1):
        warper = cv.PyRotationWarper(self.warper_type, self.scale * aspect)
        _, warped_image = warper.warp(
            img,
            Warper.get_K(camera, aspect),
            camera.R,
            cv.INTER_LINEAR,
            cv.BORDER_CONSTANT,           # cv.BORDER_REFLECT,
        )
        return warped_image
# =====================================================================================
    def warp_point(self, pt, camera, aspect=1):
        print('R:', camera.R)
        warper = cv.PyRotationWarper(self.warper_type, self.scale * aspect)
        warped_pt = warper.warpPoint(
            pt,
            Warper.get_K(camera, aspect),
            camera.R,
        )
        return warped_pt

    def warp_point_my(self, pt, camera, aspect=1):      # 意欲实现python版本的warp_point，但与其结果不一致，反而mapBackward结果更接近——什么是forward？
        K, R = Warper.get_K(camera, aspect), camera.R
        T = np.array([0,0,0], dtype=np.float32).reshape((1, 3))

        k, r_kinv, k_rinv, t = self.setCameraParams(K, R, T)
        warped_pt = self.mapForward(pt[0], pt[1], r_kinv, t, self.scale*aspect)
        # warped_pt = self.mapBackward(pt[0], pt[1], self.scale*aspect, t, k_rinv)
        
        return warped_pt
    
    def warp_point_backward(self, pt, camera, aspect=1):
        pass
        # warper = cv.PyRotationWarper(self.warper_type, self.scale * aspect)
        # warped_pt = warper.warpPointBackward(
        #     pt,
        #     Warper.get_K(camera, aspect),
        #     camera.R,
        # )
        # return warped_pt
# =====================================================================================
    def setCameraParams(self, K, R, T):
        K = np.asarray(K, dtype=np.float32)
        R = np.asarray(R, dtype=np.float32)
        T = np.asarray(T, dtype=np.float32)
        assert K.shape == (3, 3) and K.dtype == np.float32
        assert R.shape == (3, 3) and R.dtype == np.float32
        assert (T.shape == (1, 3) or T.shape == (3, 1)) and T.dtype == np.float32
        k = K.flatten()
        rinv = R.T.flatten()
        r_kinv = (R @ np.linalg.inv(K)).flatten()
        k_rinv = (K @ np.linalg.inv(R)).flatten()
        t = T.flatten()
        return k, r_kinv, k_rinv, t
    
    def mapForward(self, x, y, r_kinv, t, scale):
        x_ = r_kinv[0] * x + r_kinv[1] * y + r_kinv[2]
        y_ = r_kinv[3] * x + r_kinv[4] * y + r_kinv[5]
        z_ = r_kinv[6] * x + r_kinv[7] * y + r_kinv[8]
        x_ = t[0] + x_ / z_ * (1 - t[2])
        y_ = t[1] + y_ / z_ * (1 - t[2])
        u = scale * x_
        v = scale * y_
        return u, v

    def mapBackward(self, u, v, scale, t, k_rinv):
        u = u / scale - t[0]
        v = v / scale - t[1]
        x = k_rinv[0] * u + k_rinv[1] * v + k_rinv[2] * (1 - t[2])
        y = k_rinv[3] * u + k_rinv[4] * v + k_rinv[5] * (1 - t[2])
        z = k_rinv[6] * u + k_rinv[7] * v + k_rinv[8] * (1 - t[2])
        x /= z
        y /= z
        return x, y
# =====================================================================================

    def create_and_warp_masks(self, sizes, cameras, aspect=1):
        for size, camera in zip(sizes, cameras):
            yield self.create_and_warp_mask(size, camera, aspect)

    def create_and_warp_mask(self, size, camera, aspect=1):
        warper = cv.PyRotationWarper(self.warper_type, self.scale * aspect)
        mask = 255 * np.ones((size[1], size[0]), np.uint8)
        _, warped_mask = warper.warp(
            mask,
            Warper.get_K(camera, aspect),
            camera.R,
            cv.INTER_NEAREST,
            cv.BORDER_CONSTANT,
        )
        return warped_mask

    def warp_rois(self, sizes, cameras, aspect=1):
        roi_corners = []
        roi_sizes = []
        for size, camera in zip(sizes, cameras):
            roi = self.warp_roi(size, camera, aspect)
            roi_corners.append(roi[0:2])
            roi_sizes.append(roi[2:4])
        return roi_corners, roi_sizes

    def warp_roi(self, size, camera, aspect=1):
        warper = cv.PyRotationWarper(self.warper_type, self.scale * aspect)
        K = Warper.get_K(camera, aspect)
        return warper.warpRoi(size, K, camera.R)

    @staticmethod
    def get_K(camera, aspect=1):
        K = camera.K().astype(np.float32)
        """ Modification of intrinsic parameters needed if cameras were
        obtained on different scale than the scale of the Images which should
        be warped """
        K[0, 0] *= aspect
        K[0, 2] *= aspect
        K[1, 1] *= aspect
        K[1, 2] *= aspect
        return K
