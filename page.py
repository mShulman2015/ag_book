import cv2
from obj import OBJ

class Page:

    # original photo info
    # self.original_photo
    # self.kps
    # self.des
    #
    # # replacement photo info
    # self.replacement
    #
    # # replecement object info
    # self.obj

    def __init__(self, info, original_dir, replacement_dir, popup_file_dir, feature_detector):
        # original photo info
        self.original_photo = cv2.imread(original_dir + info["original"])
        kp_model, des_model = feature_detector.detectAndCompute(self.original_photo, None)
        self.kps = kp_model
        self.des = des_model

        # replacement photo info
        self.replacement = cv2.imread(replacement_dir + info["replacement"])

        # replecement object info
        self.obj = OBJ(popup_file_dir + info["popup_file"]["file_name"], float(info["popup_file"]["scale"]), map(float, list(info["popup_file"]["offset"])))
