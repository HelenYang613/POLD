import cv2


class FirstImageGenerator:

    def __init__(self, imagePath):
        self.imagePath = imagePath

    def generate(self):
        cap = cv2.VideoCapture(0)
        flag, frame = cap.read()
        cv2.imwrite(self.imagePath, frame)
        cv2.waitKey(1)
        cap.release()
        cv2.destroyAllWindows()
