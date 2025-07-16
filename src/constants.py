import os
import typing

def __get_locate_classes_flat_list() -> typing.List[str]:
    locate_classes_dict = {
        0: ['walk'],
        1: ['hit', 'punch'],
        2: ['kick'],
        3: ['run', 'jog'],
        4: ['jump', 'hop', 'leap'],
        5: ['throw'],
        6: ['catch'],
        7: ['step'],
        8: ['greet'],
        9: ['dance'],
        10: ['stretch', 'yoga', 'exercise / training'],
        11: ['turn', 'spin'],
        12: ['bend'],
        13: ['stand'],
        14: ['sit'],
        15: ['kneel'],
        16: ['place something'],
        17: ['grasp object'],
        18: ['take/pick something up', 'lift something'],
        19: ['scratch', 'touching face', 'touching body parts'],
    }
    return [item for sublist in locate_classes_dict.values() for item in sublist]

DEFAULT_STRIDE = 1
DEFAULT_PROC_COUNT = 8

BABEL_20_CLASSES = []
BABEL_60_CLASSES = []
BABEL_90_CLASSES = []
BABEL_120_CLASSES = []
LOCATE_CLASSES = __get_locate_classes_flat_list()

__HUGGING_FACE_TOKEN_ENV_VAR = "HUGGING_FACE_TOKEN"
HUGGING_FACE_TOKEN: str = os.getenv(__HUGGING_FACE_TOKEN_ENV_VAR) or "UNDEFINED_TOKEN"

DEFAULT_FPS = 20
DEFAULT_THRESHOLD = 0.5
DEFAULT_SEED = 1234
DEFAULT_DEVICE = 'cuda:1'
DEFAULT_PADDING_VALUE = 0.0
DEFAULT_LOAD_FROM_CACHE_FILE = True
DEFAULT_DEBUG: bool = False
DEFAULT_HYDRA_VERSION_BASE = "1.3"
DEFAULT_HYDRA_CONFIG_PATH = "configs"

BABEL_REMOTE_DATASET_NAME = "imt-ne-ai-lab/babel-dataset"
HML3D_REMOTE_DATASET_NAME = "imt-ne-ai-lab/hml3d-dataset"

MAP_AUGMENTATION_BATCH_SIZE = 32

JOINT_NAMES = {
    "smpljoints": [
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hand",
        "right_hand",
    ],
    "guoh3djoints": [
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    ],
}

JOINTS_INFOS = {
    "smpljoints": {
        "LM": JOINT_NAMES["smpljoints"].index("left_ankle"),
        "RM": JOINT_NAMES["smpljoints"].index("right_ankle"),
        "LF": JOINT_NAMES["smpljoints"].index("left_foot"),
        "RF": JOINT_NAMES["smpljoints"].index("right_foot"),
        "LS": JOINT_NAMES["smpljoints"].index("left_shoulder"),
        "RS": JOINT_NAMES["smpljoints"].index("right_shoulder"),
        "LH": JOINT_NAMES["smpljoints"].index("left_hip"),
        "RH": JOINT_NAMES["smpljoints"].index("right_hip"),
        "njoints": len(JOINT_NAMES["smpljoints"]),
    },
    "guoh3djoints": {
        "LM": JOINT_NAMES["guoh3djoints"].index("left_ankle"),
        "RM": JOINT_NAMES["guoh3djoints"].index("right_ankle"),
        "LF": JOINT_NAMES["guoh3djoints"].index("left_foot"),
        "RF": JOINT_NAMES["guoh3djoints"].index("right_foot"),
        "LS": JOINT_NAMES["guoh3djoints"].index("left_shoulder"),
        "RS": JOINT_NAMES["guoh3djoints"].index("right_shoulder"),
        "LH": JOINT_NAMES["guoh3djoints"].index("left_hip"),
        "RH": JOINT_NAMES["guoh3djoints"].index("right_hip"),
        "njoints": len(JOINT_NAMES["guoh3djoints"]),
    },
}
