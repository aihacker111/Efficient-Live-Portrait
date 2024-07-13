import torch
import numpy as np


class RetargetStitchPortrait:
    def __init__(self):
        pass

    @staticmethod
    def concat_feat(stitch_session, kp_source, kp_driving, lip_ratio, eye_ratio,
                    lip=False, eye=False) -> torch.Tensor:
        """
        kp_source: (bs, k, 3)
        kp_driving: (bs, k, 3)
        Return: (bs, 2k*3)
        """
        alert = 'batch size must be equal'
        if lip == False and eye == False:
            bs_src = kp_source.shape[0]
            bs_dri = kp_driving.shape[0]
            assert bs_src == bs_dri, alert

            feat = torch.cat([kp_source.view(bs_src, -1), kp_driving.view(bs_dri, -1)], dim=1)
            delta = stitch_session.run(None, {'input': np.array(feat)})
            return delta[0]
        elif lip == True and eye == False:
            bs_src = kp_source.shape[0]
            bs_dri = lip_ratio.shape[0]
            assert bs_src == bs_dri, alert

            feat = torch.cat([kp_source.view(bs_src, -1), lip_ratio.view(bs_dri, -1)], dim=1)
            delta_lip = stitch_session.run(None, {'input': np.array(feat)})
            return delta_lip[0]
        elif lip == False and eye == True:
            bs_src = kp_source.shape[0]
            bs_dri = eye_ratio.shape[0]
            assert bs_src == bs_dri, alert

            feat = torch.cat([kp_source.view(bs_src, -1), eye_ratio.view(bs_dri, -1)], dim=1)
            delta_eye = stitch_session.run(None, {'input': np.array(feat)})
            return delta_eye[0]

    def stitch(self, session, kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
        """
        kp_source: BxNx3
        kp_driving: BxNx3
        Return: Bx(3*num_kp+2)
        """
        delta = self.concat_feat(session, kp_source, kp_driving, lip_ratio=None, eye_ratio=None)

        return delta

    def retarget_lip(self, session, kp_source, lip_close_ratio) -> torch.Tensor:
        """
        kp_source: BxNx3
        lip_close_ratio: Bx2
        """
        delta_lip = self.concat_feat(stitch_session=session, kp_source=kp_source, kp_driving=None,
                                     lip_ratio=lip_close_ratio, eye_ratio=None, lip=True)

        return delta_lip

    def retarget_eye(self, session, kp_source, eye_close_ratio) -> torch.Tensor:
        """
        kp_source: BxNx3
        lip_close_ratio: Bx2
        """
        delta_lip = self.concat_feat(stitch_session=session, kp_source=kp_source, kp_driving=None, lip_ratio=None,
                                     eye_ratio=eye_close_ratio, eye=True)

        return delta_lip

    def stitching(self, session, kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
        """ conduct the stitching
        kp_source: Bxnum_kpx3
        kp_driving: Bxnum_kpx3
        """

        bs, num_kp = kp_source.shape[:2]

        kp_driving_new = kp_driving.clone()
        delta = self.stitch(session, kp_source, kp_driving_new)

        delta_exp = delta[..., :3 * num_kp].reshape(bs, num_kp, 3)  # 1x20x3
        delta_tx_ty = delta[..., 3 * num_kp:3 * num_kp + 2].reshape(bs, 1, 2)  # 1x1x2

        kp_driving_new += delta_exp
        kp_driving_new[..., :2] += delta_tx_ty

        return kp_driving_new

    def calculate_distance_ratio(self, lmk: np.ndarray, idx1: int, idx2: int, idx3: int, idx4: int,
                                 eps: float = 1e-6) -> np.ndarray:
        """
        Calculate the ratio of the distance between two pairs of landmarks.

        Parameters:
        lmk (np.ndarray): Landmarks array of shape (B, N, 2).
        idx1, idx2, idx3, idx4 (int): Indices of the landmarks.
        eps (float): Small value to avoid division by zero.

        Returns:
        np.ndarray: Calculated distance ratio.
        """
        return (np.linalg.norm(lmk[:, idx1] - lmk[:, idx2], axis=1, keepdims=True) /
                (np.linalg.norm(lmk[:, idx3] - lmk[:, idx4], axis=1, keepdims=True) + eps))

    def calc_eye_close_ratio(self, lmk: np.ndarray, target_eye_ratio: np.ndarray = None) -> np.ndarray:
        """
        Calculate the eye-close ratio for left and right eyes.

        Parameters:
        lmk (np.ndarray): Landmarks array of shape (B, N, 2).
        target_eye_ratio (np.ndarray, optional): Additional target eye ratio array to include.

        Returns:
        np.ndarray: Concatenated eye-close ratios.
        """
        lefteye_close_ratio = self.calculate_distance_ratio(lmk, 6, 18, 0, 12)
        righteye_close_ratio = self.calculate_distance_ratio(lmk, 30, 42, 24, 36)
        if target_eye_ratio is not None:
            return np.concatenate([lefteye_close_ratio, righteye_close_ratio, target_eye_ratio], axis=1)
        else:
            return np.concatenate([lefteye_close_ratio, righteye_close_ratio], axis=1)

    def calc_lip_close_ratio(self, lmk: np.ndarray) -> np.ndarray:
        """
        Calculate the lip-close ratio.

        Parameters:
        lmk (np.ndarray): Landmarks array of shape (B, N, 2).

        Returns:
        np.ndarray: Calculated lip-close ratio.
        """
        return self.calculate_distance_ratio(lmk, 90, 102, 48, 66)
