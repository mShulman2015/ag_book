import numpy as np
import cv2
from obj import OBJ

class Page:

    # original photo info
    # self.original_photo
    # self.kps
    # self.des
    #
    # # replacement photo info
    # self.replacement_photo
    # self.replacement_to_real_transform
    #
    # # replecement object info
    # self.obj

    def __init__(self, info, original_dir, replacement_dir, popup_file_dir, feature_detector):
        # original photo info
        self.original_photo = cv2.imread(original_dir + info["original"])
        if self.original_photo is not None:
            print("loaded original: {} - {}".format(original_dir + info["original"], self.original_photo.shape))
        else:
            print("failed to load original: {}".format(original_dir + info["original"]))
        kp_model, des_model = feature_detector.detectAndCompute(self.original_photo, None)
        self.kps = kp_model
        self.des = des_model

        # replacement photo info
        self.replacement_photo = cv2.imread(replacement_dir + info["replacement"])
        if self.replacement_photo is not None:
            print("loaded replacement: {} - {}".format(replacement_dir + info["replacement"], self.replacement_photo.shape))
        else:
            print("failed to load replacement: {}".format(replacement_dir + info["replacement"]))

        # find the transform form the replacement photo to the one we're looking for
        original_shape = self.original_photo.shape
        replace_shape = self.replacement_photo.shape
        original_points = np.array([[0, 0], [0, original_shape[0]], [original_shape[1], 0], [original_shape[1], original_shape[0]]])
        replace_points = np.array([[0, 0], [0, replace_shape[0]], [replace_shape[1], 0], [replace_shape[1], replace_shape[0]]])
        self.replacement_to_real_transform = cv2.findHomography(replace_points, original_points, cv2.RANSAC)[0]

        # replecement object info
        self.obj = OBJ(popup_file_dir + info["popup_file"]["file_name"], float(info["popup_file"]["scale"]), map(float, list(info["popup_file"]["offset"])))
