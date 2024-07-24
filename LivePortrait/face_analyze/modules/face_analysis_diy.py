# coding: utf-8

"""
face detectoin and alignment using InsightFace
"""

import numpy as np
from LivePortrait.face_analyze.utils.face_dict import Face
from LivePortrait.face_analyze.modules.scrfd import SCRFD
from LivePortrait.face_analyze.modules.landmark_2d106 import Landmark
from LivePortrait.face_analyze.modules.arc_face import ArcFaceONNX


def sort_by_direction(faces, direction: str = 'large-small', face_center=None):
    if len(faces) <= 0:
        return faces

    if direction == 'left-right':
        return sorted(faces, key=lambda face: face['bbox'][0])
    if direction == 'right-left':
        return sorted(faces, key=lambda face: face['bbox'][0], reverse=True)
    if direction == 'top-bottom':
        return sorted(faces, key=lambda face: face['bbox'][1])
    if direction == 'bottom-top':
        return sorted(faces, key=lambda face: face['bbox'][1], reverse=True)
    if direction == 'small-large':
        return sorted(faces, key=lambda face: (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1]))
    if direction == 'large-small':
        return sorted(faces, key=lambda face: (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1]),
                      reverse=True)
    if direction == 'distance-from-retarget-face':
        return sorted(faces, key=lambda face: (((face['bbox'][2] + face['bbox'][0]) / 2 - face_center[0]) ** 2 + (
                (face['bbox'][3] + face['bbox'][1]) / 2 - face_center[1]) ** 2) ** 0.5)
    return faces


class ModelRouter:
    def __init__(self, det_path, rec_path, landmark_106_path):
        self.model_det = SCRFD(det_path)
        self.model_rec = ArcFaceONNX(rec_path)
        self.model_landmark_106 = Landmark(landmark_106_path)
        self.models = self.router()

    def router(self):
        models = {
            'detection': self.model_det,
            'landmark_106': self.model_landmark_106
        }
        return models


class FaceAnalysis(ModelRouter):
    def __init__(self, det_path, rec_path, landmark_106_path, **kwargs):
        super().__init__(det_path, rec_path, landmark_106_path)
        self.det_model = self.models['detection']

    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        assert det_size is not None
        for task_name, model in self.models.items():
            if task_name == 'detection':
                model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)

    def get_detector(self, img_bgr, **kwargs):
        max_num = kwargs.get('max_num', 0)  # the number of the detected faces, 0 means no limit
        flag_do_landmark_2d_106 = kwargs.get('flag_do_landmark_2d_106', True)  # whether to do 106-point detection
        direction = kwargs.get('direction', 'large-small')  # sorting direction
        face_center = None

        bboxes, kpss = self.det_model.detect(img_bgr, max_num=max_num, metric='default')
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for task_name, model in self.models.items():
                if task_name == 'detection':
                    continue

                if (not flag_do_landmark_2d_106) and task_name == 'landmark_2d_106':
                    continue

                model.get(img_bgr, face)
            ret.append(face)

        ret = sort_by_direction(ret, direction, face_center)
        return ret

    def get_face_id(self, ret_ref, face_db):
        if ret_ref is None:
            return face_db
        # List to store indices of matching faces
        matching_indices = []

        # Compare each face in the database
        for i, face in enumerate(face_db):
            face_key = f'face_{i}'
            face_data = face[face_key]

            # Extract features for reference and current face
            feat_ref = self.model_rec.get(ret_ref['img_crop'])
            feat_db = self.model_rec.get(face_data['img_crop'])

            # Compute similarity
            sim = self.model_rec.compute_sim(feat_ref, feat_db)
            if sim >= 0.28:
                matching_indices.append(i)
        # Filter face_db to include only matching faces if there are any matches
        if matching_indices:
            filtered_face_db = [face_db[i] for i in matching_indices]
            return filtered_face_db
        else:
            return face_db  # Return the original face_db if no matches are found

    def warmup(self):
        img_bgr = np.zeros((512, 512, 3), dtype=np.uint8)
        self.get_detector(img_bgr)
