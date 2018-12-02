import numpy as np
import json
import cv2
from page import Page
import math

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

        self.camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])

        for info in data["book_sheet_defenitions"]:
            self.pages.append(Page(info, self.original_dir, self.replacement_dir, self.popup_file_dir, self.detector))

    # compute all the info we ned to find the location of the pages in the frame
    def compute_page_location_info(self, gray):
        # find keypoints in fame we're looking at
        kps, des = self.detector.detectAndCompute(gray, None)

        # find the best match for an image that we can see on a screen
        # TODO: replace this with a list of all matches that are good enough
        best_matches = None
        best_hom_transform = None
        best_index = 0
        best_value = 0
        for i in range(len(self.pages)):
            # compute the key point matches
            matches = self.matcher.match(self.pages[i].des, des)
            matches = sorted(matches, key=lambda x: x.distance)

            # attempt to find a comography taht will work for the key pionts we're considering
            src_pts = np.float32([self.pages[i].kps[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kps[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            hom_transform, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # certinty is number of inliers in homography
            certinty = sum(mask)
            # maximal check
            if certinty > self.certinty_threshold and certinty > best_value:
                best_matches = matches
                best_hom_transform = hom_transform
                best_index = i
                best_value = certinty

        return (kps, best_matches, best_hom_transform, best_index)

    # using the transfromation place the 3D object into the image
    def compute_final_frame(self, frame, page_location_info):
        kps, matches, hom_transform, best_index = page_location_info

        # if we've failed to find a match return original image
        if matches is None:
            return frame

        # ouput option 1
        # show matches
        ###########################################################################################################
        # num_to_show = min(self.best_x_matches, len(matches))
        # return cv2.drawMatches(self.pages[best_index].original_photo, self.pages[best_index].kps, frame, kps, matches[:num_to_show], 0, flags=2)
        ###########################################################################################################

        # ouput option 2
        # show outline of transform
        ###########################################################################################################
        # h, w = self.pages[best_index].original_photo.shape[:2]
        # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # # project corners into frame
        # dst = cv2.perspectiveTransform(pts, hom_transform)
        # # connect them with lines
        # return cv2.polylines(frame, [np.int32(dst)], True, best_index * 255.0 / len(self.pages[1:]), 3, cv2.LINE_AA)
        ###########################################################################################################

        # output option 3
        # show the image we are trying to overlay in the right location
        ###########################################################################################################
        # transform = np.matmul(hom_transform, self.pages[best_index].replacement_to_real_transform)
        # res_img = cv2.warpPerspective(self.pages[best_index].replacement_photo, transform, (frame.shape[1], frame.shape[0]))
        # res_mask = np.where(res_img > 0, 0, 1)
        # return np.multiply(res_mask.astype(np.uint8), frame) + res_img
        ###########################################################################################################

        # output option 4
        # project the obj onto the target image
        ###########################################################################################################
        projection = self.projection_matrix(self.camera_parameters, hom_transform)
        return self.render(frame, self.pages[best_index].obj, projection, self.pages[best_index].original_photo.shape[:2])
        ###########################################################################################################

    def projection_matrix(self, camera_parameters, homography):
        """
        From the camera calibration matrix and the estimated homography
        compute the 3D projection matrix
        """
        # Compute rotation along the x and y axis as well as the translation
        homography = homography * (-1)
        rot_and_transl = np.matmul(np.linalg.inv(camera_parameters), homography)
        col_1 = rot_and_transl[:, 0]
        col_2 = rot_and_transl[:, 1]
        col_3 = rot_and_transl[:, 2]
        # normalise vectors
        l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
        rot_1 = col_1 / l
        rot_2 = col_2 / l
        translation = col_3 / l
        # compute the orthonormal basis
        c = rot_1 + rot_2
        p = np.cross(rot_1, rot_2)
        d = np.cross(c, p)
        rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
        rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
        rot_3 = np.cross(rot_1, rot_2)
        # finally, compute the 3D projection matrix from the model to the current frame
        projection = np.stack((rot_1, rot_2, rot_3, translation)).T
        return np.dot(camera_parameters, projection)

    def render(self, frame, obj, projection, model_shape):
        """
        Render a loaded obj model into the current video frame
        """
        h, w = model_shape

        for face in obj.faces:
            points = np.array([obj.vertices[vertex - 1] * obj.scale for vertex in face[0]])
            # render model in the middle of the reference surface. To do so,
            # model points must be displaced
            points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points]) # TODO: off load this computation
            dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection).astype(np.int32)
            cv2.fillConvexPoly(frame, dst, (137, 27, 211))

        return frame
