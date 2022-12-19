import cv2 as open_cv
import numpy as np
import logging
from drawing_utils import draw_contours
from colors import COLOR_GREEN, COLOR_WHITE, COLOR_BLUE
import os
from txt2coordinate import t2c
from yml2coordinate import y2c
from inter_area import intersection


class MotionDetector:
    LAPLACIAN = 1.4
    DETECT_DELAY = 1

    def __init__(self, coordinates, start_frame):
        self.coordinates_data = coordinates
        self.start_frame = start_frame
        self.contours = []
        self.bounds = []
        self.mask = []

    def empty(path):
        if not os.listdir(path): 
            return True
        else:
            return False

    def detect_motion(self):
        capture = open_cv.VideoCapture(0)
        capture.set(open_cv.CAP_PROP_POS_FRAMES, self.start_frame)

        coordinates_data = self.coordinates_data
        logging.debug("coordinates data: %s", coordinates_data)

        for p in coordinates_data:
            coordinates = self._coordinates(p)
            logging.debug("coordinates: %s", coordinates)

            rect = open_cv.boundingRect(coordinates)
            logging.debug("rect: %s", rect)

            new_coordinates = coordinates.copy()
            new_coordinates[:, 0] = coordinates[:, 0] - rect[0]
            new_coordinates[:, 1] = coordinates[:, 1] - rect[1]
            logging.debug("new_coordinates: %s", new_coordinates)

            self.contours.append(coordinates)
            self.bounds.append(rect)

            mask = open_cv.drawContours(
                np.zeros((rect[3], rect[2]), dtype=np.uint8),
                [new_coordinates],
                contourIdx=-1,
                color=255,
                thickness=-1,
                lineType=open_cv.LINE_8)

            mask = mask == 255
            self.mask.append(mask)
            logging.debug("mask: %s", self.mask)

        statuses = [False] * len(coordinates_data)
        times = [None] * len(coordinates_data)

        while capture.isOpened():
            result, frame = capture.read()
            if frame is None:
                break

            if not result:
                raise CaptureReadError("Error reading video capture on frame %s" % str(frame))

            blurred = open_cv.GaussianBlur(frame.copy(), (5, 5), 3)
            grayed = open_cv.cvtColor(blurred, open_cv.COLOR_BGR2GRAY)
            new_frame = frame.copy()
            logging.debug("new_frame: %s", new_frame)

            position_in_seconds = capture.get(open_cv.CAP_PROP_POS_MSEC) / 1000.0
            label_path = '/home/yangpeng/yolov5/runs/detect/exp8/labels'
            yaml_path = '/home/yangpeng/Desktop/parking_2D/data/coordinates.yml'

            if empty(label_path) == True:
                for index, c in enumerate(coordinates_data):
                    status = self.__apply(grayed, index, c)

                    if times[index] is not None and self.same_status(statuses, index, status):
                        times[index] = None
                        continue

                    if times[index] is not None and self.status_changed(statuses, index, status):
                        if position_in_seconds - times[index] >= MotionDetector.DETECT_DELAY:
                            statuses[index] = status
                            times[index] = None
                        continue

                    if times[index] is None and self.status_changed(statuses, index, status):
                        times[index] = position_in_seconds
            else:
                bbox = t2c(label_path)
                space = y2c(yaml_path)
                occu = [False] * len(space)
                for i in range(len(space)):
                    p1 = space[i].reshape([4,2])
                    size = intersection(p1, p1)
                    flag = 0
                    for j in range(len(bbox)):
                        p2 = [(bbox[j,1], bbox[j,2]), (bbox[j,3], bbox[j,2]), (bbox[j,3], bbox[j,4]), (bbox[j,1], bbox[j,4])]
                        inter_area = intersection(p1, p2)
                        if inter_area >= size * 0.6:
                            flag = 1
                            break
                    if flag == 1:
                        occu[i] = True
                        continue

            for index, p in enumerate(coordinates_data):
                coordinates = self._coordinates(p)

                color = COLOR_GREEN if (statuses[index] & occu[index]) | occu[index] else COLOR_BLUE
                draw_contours(new_frame, coordinates, str(p["id"] + 1), COLOR_WHITE, color)

            open_cv.imshow('real-time', new_frame)
            k = open_cv.waitKey(1)
            if k == ord("q"):
                break
        capture.release()
        open_cv.destroyAllWindows()

    def __apply(self, grayed, index, p):
        coordinates = self._coordinates(p)
        logging.debug("points: %s", coordinates)

        rect = self.bounds[index]
        logging.debug("rect: %s", rect)

        roi_gray = grayed[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]
        laplacian = open_cv.Laplacian(roi_gray, open_cv.CV_64F)
        logging.debug("laplacian: %s", laplacian)

        coordinates[:, 0] = coordinates[:, 0] - rect[0]
        coordinates[:, 1] = coordinates[:, 1] - rect[1]

        status = np.mean(np.abs(laplacian * self.mask[index])) < MotionDetector.LAPLACIAN
        logging.debug("status: %s", status)

        return status

    @staticmethod
    def _coordinates(p):
        return np.array(p["coordinates"])

    @staticmethod
    def same_status(coordinates_status, index, status):
        return status == coordinates_status[index]

    @staticmethod
    def status_changed(coordinates_status, index, status):
        return status != coordinates_status[index]


class CaptureReadError(Exception):
    pass
