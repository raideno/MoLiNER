import torch
import typing

import numpy as np

class MotionNormalizer:
    """
    Loads and applies normalization statistics for motion data.
    """
    def __init__(self, stats_path: str):
        stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        self.means = stats["mean"]
        self.stds = stats["std"]

    def normalize(self, motion: typing.Union[typing.Dict[str, typing.Any], np.ndarray, list]) -> typing.Union[typing.Dict[str, typing.Any], np.ndarray, list]:
        """
        Normalize all present motion types using loaded mean and std.
        """
        if isinstance(motion, dict):
            normed = motion.copy()
            if "new_joint_vecs" in motion and "new_joint_vecs" in self.means:
                arr = np.array(motion["new_joint_vecs"])
                normed["new_joint_vecs"] = ((arr - self.means["new_joint_vecs"]) / self.stds["new_joint_vecs"]).tolist()
                
            if "new_joints" in motion and "new_joints" in self.means:
                arr = np.array(motion["new_joints"])
                normed["new_joints"] = ((arr - self.means["new_joints"]) / self.stds["new_joints"]).tolist()
                
            return normed

        if "new_joint_vecs" in self.means:
            arr = np.array(motion)
            return ((arr - self.means["new_joint_vecs"]) / self.stds["new_joint_vecs"]).tolist()
        
        if "new_joints" in self.means:
            arr = np.array(motion)
            return ((arr - self.means["new_joints"]) / self.stds["new_joints"]).tolist()
        
        return motion
