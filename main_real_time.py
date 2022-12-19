import argparse
import yaml
from coordinates_generator import CoordinatesGenerator
from first_image_generator import FirstImageGenerator
from motion_detector import MotionDetector
from colors import *
import logging
import cv2
import os


def main():
    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    image_file = args.image_file
    data_file = args.data_file
    start_frame = args.start_frame

    imageGenerator = FirstImageGenerator(image_file)
    imageGenerator.generate()

    with open(data_file, "w+") as points:
        generator = CoordinatesGenerator(image_file, points, COLOR_RED)
        generator.generate()

    with open(data_file, "r") as data:
        points = yaml.full_load(data)
        camera = cv2.VideoCapture(0)
        detector = MotionDetector(points, int(start_frame))
        detector.detect_motion()

def parse_args():
    parser = argparse.ArgumentParser(description='Generates Coordinates File')

    parser.add_argument("--image",
                        dest="image_file",
                        required=False,
                        help="Image file to generate coordinates on")

    parser.add_argument("--data",
                        dest="data_file",
                        required=True,
                        help="Data file to be used with OpenCV")

    parser.add_argument("--start-frame",
                        dest="start_frame",
                        required=False,
                        default=1,
                        help="Starting frame on the video")

    return parser.parse_args()


if __name__ == '__main__':
    os.system('python3 /home/yangpeng/yolov5/detect.py --weights yolov5s.pt --source 0')
    main()
