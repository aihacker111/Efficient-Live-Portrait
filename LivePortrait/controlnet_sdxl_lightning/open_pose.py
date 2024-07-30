from dwpose import DWposeDetector
from PIL import Image
import cv2


class OpenPose:
    def __init__(self):
        pass

    @staticmethod
    def load_image(image_path):
        image = Image.open(image_path)
        return image

    def get_pose(self, image_path):
        pose_detector = DWposeDetector()
        image = self.load_image(image_path)
        pose_image, _ = pose_detector(image)
        pose_image = pose_image[:, :, ::-1]
        return pose_image


if __name__ == '__main__':
    open_pose = OpenPose()
    pose_img = open_pose.get_pose('/Users/macbook/Downloads/Efficient-Face2Vid-Portrait/img_1.png')
    cv2.imshow('a', pose_img)
    cv2.waitKey(0)
