from types import SimpleNamespace

from .blender import Blender
from .camera_adjuster import CameraAdjuster
from .camera_estimator import CameraEstimator
from .camera_wave_corrector import WaveCorrector
from .cropper import Cropper
from .exposure_error_compensator import ExposureErrorCompensator
from .feature_detector import FeatureDetector
from .feature_matcher import FeatureMatcher
from .my_image_handler import ImageHandler
from .seam_finder import SeamFinder
from .stitching_error import StitchingError
from .subsetter import Subsetter
from .timelapser import Timelapser
from .warper import Warper

from datetime import datetime


from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
def plot_image(img, figsize_in_inches=(5,5)):
    fig, ax = plt.subplots(figsize=figsize_in_inches)
    ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()  # plt.show(block=False)
def plot_images(imgs, figsize_in_inches=(5,5)):
    fig, axs = plt.subplots(1, len(imgs), figsize=figsize_in_inches)
    for col, img in enumerate(imgs):
        axs[col].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()
    

class my_Stitcher:
    DEFAULT_SETTINGS = {
        "medium_megapix": ImageHandler.DEFAULT_MEDIUM_MEGAPIX,
        "detector": FeatureDetector.DEFAULT_DETECTOR,       # +++
        "nfeatures": 500,
        "matcher_type": FeatureMatcher.DEFAULT_MATCHER,     # +++
        "range_width": FeatureMatcher.DEFAULT_RANGE_WIDTH,
        "try_use_gpu": False,
        "match_conf": None,
        "confidence_threshold": Subsetter.DEFAULT_CONFIDENCE_THRESHOLD,
        "matches_graph_dot_file": Subsetter.DEFAULT_MATCHES_GRAPH_DOT_FILE,
        "estimator": CameraEstimator.DEFAULT_CAMERA_ESTIMATOR,      # +++
        "adjuster": CameraAdjuster.DEFAULT_CAMERA_ADJUSTER,         # +++
        "refinement_mask": CameraAdjuster.DEFAULT_REFINEMENT_MASK,
        "wave_correct_kind": WaveCorrector.DEFAULT_WAVE_CORRECTION, # +
        "warper_type": Warper.DEFAULT_WARP_TYPE,            # +++
        "low_megapix": ImageHandler.DEFAULT_LOW_MEGAPIX,
        "crop": Cropper.DEFAULT_CROP,
        "compensator": ExposureErrorCompensator.DEFAULT_COMPENSATOR,    # +
        "nr_feeds": ExposureErrorCompensator.DEFAULT_NR_FEEDS,
        "block_size": ExposureErrorCompensator.DEFAULT_BLOCK_SIZE,
        "finder": SeamFinder.DEFAULT_SEAM_FINDER,
        "final_megapix": ImageHandler.DEFAULT_FINAL_MEGAPIX,
        "blender_type": Blender.DEFAULT_BLENDER,            # +
        "blend_strength": Blender.DEFAULT_BLEND_STRENGTH,
        "timelapse": Timelapser.DEFAULT_TIMELAPSE,
        "timelapse_prefix": Timelapser.DEFAULT_TIMELAPSE_PREFIX,
    }

    def __init__(self, **kwargs):
        self.initialize_stitcher(**kwargs)

    def initialize_stitcher(self, **kwargs):
        self.settings = self.DEFAULT_SETTINGS.copy()
        self.validate_kwargs(kwargs)
        self.settings.update(kwargs)

        args = SimpleNamespace(**self.settings)
        self.img_handler = ImageHandler(
            args.medium_megapix, args.low_megapix, args.final_megapix
        )
        if args.detector in ("orb", "sift"):
            self.detector = FeatureDetector(args.detector, nfeatures=args.nfeatures)
        else:
            self.detector = FeatureDetector(args.detector)
        match_conf = FeatureMatcher.get_match_conf(args.match_conf, args.detector)
        self.matcher = FeatureMatcher(
            args.matcher_type,
            args.range_width,
            try_use_gpu=args.try_use_gpu,
            match_conf=match_conf,
        )
        self.subsetter = Subsetter(
            args.confidence_threshold, args.matches_graph_dot_file
        )
        self.camera_estimator = CameraEstimator(args.estimator)
        self.camera_adjuster = CameraAdjuster(
            args.adjuster, args.refinement_mask, args.confidence_threshold
        )
        self.wave_corrector = WaveCorrector(args.wave_correct_kind)
        self.warper = Warper(args.warper_type)
        self.cropper = Cropper(args.crop)
        self.compensator = ExposureErrorCompensator(
            args.compensator, args.nr_feeds, args.block_size
        )
        self.seam_finder = SeamFinder(args.finder)
        self.blender = Blender(args.blender_type, args.blend_strength)
        self.timelapser = Timelapser(args.timelapse, args.timelapse_prefix)

    '''
    def stitch_once(self, img_names):
        t0 = datetime.now()
        print(f'===== START ===== {t0}')

        self.initialize_registration(img_names)
        imgs = self.resize_medium_resolution()
        features = self.find_features(imgs)
        matches = self.match_features(features)
        print(f'{datetime.now()-t0} ===== 0) Matching ok. =====')

        imgs, features, matches = self.subset(imgs, features, matches)
        cameras = self.estimate_camera_parameters(features, matches)
        cameras = self.refine_camera_parameters(features, matches, cameras)
        cameras = self.perform_wave_correction(cameras)
        print(f'{datetime.now()-t0} ===== 0) Calibration ok. =====')

        self.estimate_scale(cameras)
        imgs = self.resize_low_resolution(imgs)
        imgs, masks, corners, sizes = self.warp_low_resolution(imgs, cameras)
        self.prepare_cropper(imgs, masks, corners, sizes)
        imgs, masks, corners, sizes = self.crop_low_resolution(imgs, masks, corners, sizes)
        self.estimate_exposure_errors(corners, imgs, masks)
        print(f'{datetime.now()-t0} ===== 1) Exposure errors ok. =====')

        seam_masks = self.find_seam_masks(imgs, corners, masks)
        imgs = self.resize_final_resolution()
        imgs, masks, corners, sizes = self.warp_final_resolution(imgs, cameras)
        imgs, masks, corners, sizes = self.crop_final_resolution(imgs, masks, corners, sizes)
        self.set_masks(masks)
        imgs = self.compensate_exposure_errors(corners, imgs)
        seam_masks = self.resize_seam_masks(seam_masks)
        print(f'{datetime.now()-t0} ===== 2) Seam masks ok. =====')

        self.initialize_composition(corners, sizes)
        self.blend_images(imgs, seam_masks, corners)
        print(f'{datetime.now()-t0} ===== 3) Blending ok. =====')

        pano = self.create_final_panorama()
        return pano
    '''

    '''
    # Add 20230719
    def stitch_by_mask(self, img_names, img_lst, img_masks, incre_img=None, incre_mask=None):
        origin_img_masks = img_masks
        origin_img_names = img_names
        t0 = datetime.now()
        print(f'===== START ===== {t0}')

        self.initialize_registration(img_names, incre_img, incre_mask)

        # ===== Resize medium (from origin) ===== : CALIBRATION
        # imgs = self.resize_medium_resolution(img_names=img_names)
        imgs = self.resize_medium_resolution(img_lst=img_lst)
        img_masks = self.mask_resize_medium_resolution(img_lst=origin_img_masks)
        # Feature and match
        features = self.find_features(imgs, img_masks)
        matches = self.match_features(features)
        # all_relevant_matches = self.matcher.draw_matches_matrix(imgs, features, matches)
        # for idx1, idx2, img in all_relevant_matches:
        #     print(f"Matches Image {idx1+1} to Image {idx2+1}")
        #     plot_image(img, (20,10))
        print(f'{datetime.now()-t0} ===== 0) Matching ok. =====')
        # Calibration
        # imgs, features, matches = self.subset(imgs, features, matches)
        cameras = self.estimate_camera_parameters(features, matches)
        cameras = self.refine_camera_parameters(features, matches, cameras)
        cameras = self.perform_wave_correction(cameras)
        print(f'{datetime.now()-t0} ===== 0) Calibration ok. =====')

        # ===== Resize low (from medium) ===== : 
        self.estimate_scale(cameras)
        imgs = self.resize_low_resolution(img_lst=imgs)
        img_masks = self.mask_resize_low_resolution(img_lst=img_masks)
        # warp low
        imgs, warp_masks1, corners, sizes = self.warp_low_resolution(imgs, cameras)
        warp_masks2, _, _, _ = self.warp_low_resolution(img_masks, cameras)
        warp_masks = [cv.bitwise_and(a,b) for a,b in zip(warp_masks1, warp_masks2)]
        # crop low
        self.prepare_cropper(imgs, warp_masks, corners, sizes)
        imgs, warp_masks, corners, sizes = self.crop_low_resolution(imgs, warp_masks, corners, sizes)
        # Exposure errors
        self.estimate_exposure_errors(corners, imgs, warp_masks)
        print(f'{datetime.now()-t0} ===== 1) Exposure errors ok. =====')
        # Seam masks
        seam_masks = self.find_seam_masks(imgs, corners, warp_masks)
            # seam_masks_plots = [self.seam_finder.draw_seam_mask(img, seam_mask) for img, seam_mask in zip(imgs, seam_masks)]
            # plot_images(seam_masks_plots)

        #  ===== Resize final (from origin) ===== 
        # imgs = self.resize_final_resolution(img_names=img_names)
        imgs = self.resize_final_resolution(img_lst=img_lst)
        img_masks = self.mask_resize_final_resolution(img_lst=origin_img_masks)
        # warp final
        imgs, warp_masks1, corners, sizes = self.warp_final_resolution(imgs, cameras)
        warp_masks2, warp_masks2_, _, _ = self.warp_final_resolution(img_masks, cameras)
        warp_masks = [cv.bitwise_and(a,b) for a,b in zip(warp_masks1, warp_masks2)]
        # crop final
        imgs, warp_masks, corners, sizes = self.crop_final_resolution(imgs, warp_masks, corners, sizes)
        # resize seam masks
        self.set_masks(warp_masks)
        imgs = self.compensate_exposure_errors(corners, imgs)
        seam_masks = self.resize_seam_masks(seam_masks)

        print(f'{datetime.now()-t0} ===== 2) Seam masks ok. =====')

        dst_sz = self.blender_prepare(corners, sizes)
        self.blender_feed(imgs, seam_masks, corners)
        pano, pano_mask = self.blender_blend()
        print(f'{datetime.now()-t0} ===== 3) Stitching ok. =====')
        
        return pano, pano_mask
    '''

    # Add 20230730
    def stitch_by_local_feature(self, img_names, img_lst, img_masks, matching_mask0=None, incre_img=None, incre_mask=None):
        origin_img_masks = img_masks
        origin_img_names = img_names
        t0 = datetime.now()
        print(f'===== START ===== {t0}')

        self.initialize_registration(img_names, incre_img, incre_mask)

        # ===== Resize medium (from origin) ===== : CALIBRATION
        # imgs = self.resize_medium_resolution(img_names=img_names)
        imgs = self.resize_medium_resolution(img_lst=img_lst)
        img_masks = self.mask_resize_medium_resolution(img_lst=origin_img_masks)
        # plot_images([imgs[0], imgs[1], img_masks[0], img_masks[1]])

        # Feature and match
        if matching_mask0 is not None:
            features = self.find_features(imgs, [matching_mask0, img_masks[1]])
        else:
            features = self.find_features(imgs, img_masks)
        matches = self.match_features(features)
        # print(len(matches[0].matches), len(features[0].getKeypoints()))
        # print(len(matches[1].matches), len(features[1].getKeypoints()))

        # all_relevant_matches = self.matcher.draw_matches_matrix(imgs, features, matches)
        # for idx1, idx2, img in all_relevant_matches:
        #     print(f"Matches Image {idx1+1} to Image {idx2+1}")
        #     plot_image(img, (20,10))
        print(f'{datetime.now()-t0} ===== 0) Matching ok. =====')
        # Calibration
        # imgs, features, matches = self.subset(imgs, features, matches)
        cameras = self.estimate_camera_parameters(features, matches)
        # cam0 = cameras[0]
        # cam1 = cameras[1]
        # print('R0', cam0.R.flatten())
        # print('R1', cam1.R.flatten())
        # k0 = Warper.get_K(cam0, 1)
        # k1 = Warper.get_K(cam1, 1)
        # print('k0', k0.flatten())
        # print('k1', k1.flatten())

        cameras = self.refine_camera_parameters(features, matches, cameras)
        # cameras = self.perform_wave_correction(cameras)
        print(f'{datetime.now()-t0} ===== 0) Calibration ok. =====')

        # ===== Resize low (from medium) ===== : 
        self.estimate_scale(cameras)
        imgs = self.resize_low_resolution(img_lst=imgs)
        img_masks = self.mask_resize_low_resolution(img_lst=img_masks)
        
        # warp low, crop low
        imgs, warp_masks1, corners, sizes = self.warp_low_resolution(imgs, cameras)
        warp_masks2, warp_masks2_, _, _ = self.warp_low_resolution(img_masks, cameras)
        print(f'{datetime.now()-t0} ===== 1) low mask start. =====')
        warp_masks = [cv.bitwise_and(a,b) for a,b in zip(warp_masks1, warp_masks2)]
        print(f'{datetime.now()-t0} ===== 1) low mask end. =====')

        # plot_images([imgs[0], imgs[1], warp_masks[0], warp_masks[1], 
        #              img_masks[0], img_masks[1], warp_masks2[0], warp_masks2[1]])
        # plot_images([imgs[0], imgs[1], warp_masks[0], warp_masks[1]])
        # a = []
        # print('corners',corners)
        # print('sizes',sizes)
        # plot_images(a)
        # imgs=imgs_

            # self.prepare_cropper(imgs, warp_masks, corners, sizes)
        imgs, warp_masks, corners, sizes = self.crop_low_resolution(imgs, warp_masks, corners, sizes)
        # Exposure errors
        self.estimate_exposure_errors(corners, imgs, warp_masks)
        print(f'{datetime.now()-t0} ===== 1) Exposure errors ok. =====')
        # Seam masks
        seam_masks = self.find_seam_masks(imgs, corners, warp_masks)
            # seam_masks_plots = [self.seam_finder.draw_seam_mask(img, seam_mask) for img, seam_mask in zip(imgs, seam_masks)]
            # plot_images(seam_masks_plots)
        print(f'{datetime.now()-t0} ===== 2) Seam masks finding ok. =====')

        #  ===== Resize final (from origin) ===== 
        # imgs = self.resize_final_resolution(img_names=img_names)
        imgs = self.resize_final_resolution(img_lst=img_lst)
        img_masks = self.mask_resize_final_resolution(img_lst=origin_img_masks)
        # warp final, crop final
        print(f'{datetime.now()-t0} ===== 2) Resize final ok. =====')
        imgs, warp_masks1, corners, sizes = self.warp_final_resolution(imgs, cameras)
        warp_masks1 = list(warp_masks1)
        print(f'{datetime.now()-t0} ===== 2) Warp img ok. =====')
        warp_masks2, warp_masks2_, _, _ = self.warp_final_resolution(img_masks, cameras)    
        warp_masks2 = list(warp_masks2)

        print(f'{datetime.now()-t0} ===== 2) final mask start. =====')
        warp_masks = [cv.bitwise_and(a,b) for a,b in zip(warp_masks1, warp_masks2)]
        print(f'{datetime.now()-t0} ===== 2) final mask end. =====')

        # imgs = list(imgs_)
        # warp_masks = list(warp_masks)
        # plot_images([imgs[0], imgs[1], warp_masks[0], warp_masks[1]])
        # print('shapes', imgs[0].shape, imgs[1].shape)
        # print('corners',corners)
        # print('sizes',sizes)
        # print()
        # imgs=imgs_

        # crop final
        imgs, warp_masks, corners, sizes = self.crop_final_resolution(imgs, warp_masks, corners, sizes)
        # resize seam masks
        self.set_masks(warp_masks)
        imgs = self.compensate_exposure_errors(corners, imgs)
        seam_masks = self.resize_seam_masks(seam_masks)
        # seam_masks_plots = [SeamFinder.draw_seam_mask(img, seam_mask) for img, seam_mask in zip(imgs, seam_masks)]
        # plot_images(seam_masks_plots, (15,10))
        # print(seam_masks_[0].get().shape, seam_masks_[1].get().shape)

        print(f'{datetime.now()-t0} ===== 2) Seam masks ok. =====')

        dst_sz = self.blender_prepare(corners, sizes)
        print(f'{datetime.now()-t0} ===== 3) Blendering preparing ok. =====')
        self.blender_feed(imgs, seam_masks, corners)
        print(f'{datetime.now()-t0} ===== 3) Blendering feeding ok. =====')
        pano, pano_mask = self.blender_blend()
        print(f'{datetime.now()-t0} ===== 3) Blendering blending ok. =====')
        
        return pano, pano_mask, corners, sizes, dst_sz



    # Add 20230719
    def mask_resize_medium_resolution(self, img_lst=None, img_names=None):
        return list(self.img_handler.mask_resize_to_medium_resolution(img_lst=img_lst, img_names=img_names))
    def mask_resize_low_resolution(self, img_lst):
        return list(self.img_handler.mask_resize_to_low_resolution(img_lst))
    def mask_resize_final_resolution(self, img_lst=None, img_names=None):
        return self.img_handler.mask_resize_to_final_resolution(img_lst=img_lst, img_names=img_names)

    # Modify 230719
    def resize_medium_resolution(self, img_lst=None, img_names=None):
        if (img_lst is None and img_names is None) or (img_lst is not None and img_names is not None):
            raise StitchingError('Either img_lst or img_names should be written...')
        return list(self.img_handler.resize_to_medium_resolution(img_lst=img_lst, img_names=img_names))
    def resize_low_resolution(self, img_lst):
        return list(self.img_handler.resize_to_low_resolution(img_lst))
    def resize_final_resolution(self, img_lst=None, img_names=None):
        if (img_lst is None and img_names is None) or (img_lst is not None and img_names is not None):
            raise StitchingError('Either img_lst or img_names should be written...')
        return self.img_handler.resize_to_final_resolution(img_lst=img_lst, img_names=img_names)

    def initialize_registration(self, img_names, incre_img=None, incre_mask=None):
        self.img_handler.set_incre_img(incre_img)           # Add 20230719
        self.img_handler.set_incre_mask(incre_mask)         # Add 20230719
        self.img_handler.set_img_names(img_names)

    def find_features(self, imgs, masks=None):
        return [self.detector.detect_features(img, mask) for img,mask in zip(imgs,masks)]

    def match_features(self, features):
        return self.matcher.match_features(features)

    def subset(self, imgs, features, matches):
        names, sizes, imgs, features, matches = self.subsetter.subset(
            self.img_handler.img_names,
            self.img_handler.img_sizes,
            imgs,
            features,
            matches,
        )
        self.img_handler.img_names, self.img_handler.img_sizes = names, sizes
        return imgs, features, matches

    def estimate_camera_parameters(self, features, matches):
        return self.camera_estimator.estimate(features, matches)

    def refine_camera_parameters(self, features, matches, cameras):
        return self.camera_adjuster.adjust(features, matches, cameras)

    def perform_wave_correction(self, cameras):
        return self.wave_corrector.correct(cameras)

    def estimate_scale(self, cameras):
        self.warper.set_scale(cameras)

    def warp_low_resolution(self, imgs, cameras):
        sizes = self.img_handler.get_low_img_sizes()
        camera_aspect = self.img_handler.get_medium_to_low_ratio()
        imgs, masks, corners, sizes = self.warp(imgs, cameras, sizes, camera_aspect)
        return list(imgs), list(masks), corners, sizes

    def warp_final_resolution(self, imgs, cameras):
        sizes = self.img_handler.get_final_img_sizes()
        camera_aspect = self.img_handler.get_medium_to_final_ratio()
        return self.warp(imgs, cameras, sizes, camera_aspect)

    def warp(self, imgs, cameras, sizes, aspect=1):
        # imgs=list(imgs)
        # if len(imgs[1].shape)==3:
        #     h,w,_ = imgs[1].shape
        # elif len(imgs[1].shape)==2:
        #     h,w = imgs[1].shape
        
        # warped_pt = self.warper.warp_point([0,0], cameras[1], aspect)
        # print('warped_pt2[0,0]', warped_pt)
        # warped_pt = self.warper.warp_point_my([0,0], cameras[1], aspect)
        # print('my_warped_pt2[0,0]---', warped_pt)      
        # warped_pt = self.warper.warp_point([w,0], cameras[1], aspect)
        # print('warped_pt2[w,0]', warped_pt)
        # warped_pt = self.warper.warp_point_my([w,0], cameras[1], aspect)
        # print('my_warped_pt2[w,0]---', warped_pt)

        imgs = self.warper.warp_images(imgs, cameras, aspect)
        masks = self.warper.create_and_warp_masks(sizes, cameras, aspect)
        corners, sizes = self.warper.warp_rois(sizes, cameras, aspect)
        
        # print('corner:', corners[1])          corners是self.warper.warp_point([0,0])和[w,0]得来的，原因待探究

        return imgs, masks, corners, sizes

    def prepare_cropper(self, imgs, masks, corners, sizes):
        self.cropper.prepare(imgs, masks, corners, sizes)

    def crop_low_resolution(self, imgs, masks, corners, sizes):
        imgs, masks, corners, sizes = self.crop(imgs, masks, corners, sizes)
        return list(imgs), list(masks), corners, sizes

    def crop_final_resolution(self, imgs, masks, corners, sizes):
        lir_aspect = self.img_handler.get_low_to_final_ratio()
        return self.crop(imgs, masks, corners, sizes, lir_aspect)

    def crop(self, imgs, masks, corners, sizes, aspect=1):
        masks = self.cropper.crop_images(masks, aspect)
        imgs = self.cropper.crop_images(imgs, aspect)
        corners, sizes = self.cropper.crop_rois(corners, sizes, aspect)
        return imgs, masks, corners, sizes

    def estimate_exposure_errors(self, corners, imgs, masks):
        self.compensator.feed(corners, imgs, masks)

    def find_seam_masks(self, imgs, corners, masks):
        return self.seam_finder.find(imgs, corners, masks)

    def compensate_exposure_errors(self, corners, imgs):
        for idx, (corner, img) in enumerate(zip(corners, imgs)):
            yield self.compensator.apply(idx, corner, img, self.get_mask(idx))

    def resize_seam_masks(self, seam_masks):
        for idx, seam_mask in enumerate(seam_masks):
            yield SeamFinder.resize(seam_mask, self.get_mask(idx))

    def set_masks(self, mask_generator):
        self.masks = mask_generator
        self.mask_index = -1

    def get_mask(self, idx):
        if idx == self.mask_index + 1:
            self.mask_index += 1
            self.mask = next(self.masks)
            return self.mask
        elif idx == self.mask_index:
            return self.mask
        else:
            raise StitchingError("Invalid Mask Index!")

    def blender_prepare(self, corners, sizes):
        if self.timelapser.do_timelapse:
            self.timelapser.initialize(corners, sizes)
        else:
            dst_sz = self.blender.prepare(corners, sizes)
            return dst_sz

    def blender_feed(self, imgs, masks, corners):
        for idx, (img, mask, corner) in enumerate(zip(imgs, masks, corners)):
            if self.timelapser.do_timelapse:
                self.timelapser.process_and_save_frame(
                    self.img_handler.img_names[idx], img, corner
                )
            else:
                # plot_images([img, mask.get()])
                self.blender.feed(img, mask, corner)

    def blender_blend(self):
        if not self.timelapser.do_timelapse:
            panorama, panorama_mask = self.blender.blend()
            return panorama, panorama_mask

    def validate_kwargs(self, kwargs):
        for arg in kwargs:
            if arg not in self.DEFAULT_SETTINGS:
                raise StitchingError("Invalid Argument: " + arg)

    
class my_AffineStitcher(my_Stitcher):
    AFFINE_DEFAULTS = {
        "estimator": "affine",
        "wave_correct_kind": "no",
        "matcher_type": "affine",
        "adjuster": "affine",
        "warper_type": "affine",
        "compensator": "no",
        "crop": False
    }

    DEFAULT_SETTINGS = my_Stitcher.DEFAULT_SETTINGS.copy()
    DEFAULT_SETTINGS.update(AFFINE_DEFAULTS)
