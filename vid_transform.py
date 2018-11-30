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
        self.best_x_matches = 30 # number of best matches to keep when displaying result
        self.certinty_threshold = 20 # number of inliers we require

        for info in data["book_sheet_defenitions"]:
            self.pages.append(Page(info, self.original_dir, self.replacement_dir, self.popup_file_dir, self.detector))

    # compute all the info we ned to find the location of the pages in the frame
    def compute_page_location_info(self, gray):
        # find keypoints in fame we're looking at
        kps, des = self.detector.detectAndCompute(gray, None)

        # find the best match for an image that we can see on a screen
        # TODO: replace this with a list of all matches that are good enough
        best_matches = None
        best_M = None
        best_index = 0
        best_value = 0
        for i in range(len(self.pages)):
            # compute the key point matches
            matches = self.matcher.match(self.pages[i].des, des)
            matches = sorted(matches, key=lambda x: x.distance)

            # attempt to find a comography taht will work for the key pionts we're considering
            src_pts = np.float32([self.pages[i].kps[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kps[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # certinty is number of inliers in homography
            certinty = sum(mask)
            # maximal check
            if certinty > self.certinty_threshold and certinty > best_value:
                best_matches = matches
                best_M = M
                best_index = i
                best_value = certinty

        return (kps, best_matches, best_M, best_index)

    # using the transfromation place the 3D object into the image
    def compute_final_frame(self, frame, page_location_info):
        kps, matches, M, best_index = page_location_info

        # if we've failed to find a match return original image
        if matches is None:
            return frame

        # ouput option 1
        # show matches
        # num_to_show = min(self.best_x_matches, len(matches))
        # return cv2.drawMatches(self.pages[best_index].original_photo, self.pages[best_index].kps, frame, kps, matches[:num_to_show], 0, flags=2)

        # ouput option 2
        # show outline of transform
        h, w = self.pages[best_index].original_photo.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # project corners into frame
        dst = cv2.perspectiveTransform(pts, M)
        # connect them with lines
        return cv2.polylines(frame, [np.int32(dst)], True, best_index * 255.0 / len(self.pages[1:]), 3, cv2.LINE_AA)

        # output option 3
        # show the image we are trying to overlay in the right location
