import numpy as np
import json
import cv2
from page import Page

class Transformer:
    def __init__(self, booket_defenition_file):
        with open(booket_defenition_file, 'r') as f:
            data = json.load(f)
        self.original_dir = data["original_dir"]
        self.replacement_dir = data["replacement_dir"]
        self.popup_file_dir = data["popup_file_dir"]
        self.pages = []

        self.detector = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        for info in data["book_sheet_defenitions"]:
            self.pages.append(Page(info, self.original_dir, self.replacement_dir, self.popup_file_dir, self.detector))

    # compute all the info we ned to find the location of the pages in the frame
    def compute_page_location_info(self, gray):
        kps, des = self.detector.detectAndCompute(gray, None)
        return kps
    def compute_page_locatoin_frame(self, gray, page_location_info):
        return cv2.drawKeypoints(gray, page_location_info, gray)

    # compute the transform to a new image onto the old image locatoin
    def compute_flat_transform(self, gray, page_location_info):
        return None
    def compute_flat_frame(self, frame, flat_transform):
        return frame

    # using the transfromation place the 3D object into the image
    def compute_final_frame(self, flat_frame, flat_transform):
        return flat_frame
