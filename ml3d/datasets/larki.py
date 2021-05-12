import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from larki_pc.io import e57

from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import DATASET

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Larki(BaseDataset):
    def __init__(
        self,
        dataset_path: str,
        name: str = "Larki",
        cache_dir: str = ".logs/cache",
        use_cache: bool = False,
        class_weights: List[int] = [
            55437630, 320797, 541736, 2578735, 3274484, 552662, 184064,
            78858, 240942562, 17294618, 170599734, 6369672, 230413074,
            101130274, 476491114, 9833174, 129609852, 4506626, 1168181
        ],
        ignored_label_inds: List[int] = [0],
        test_result_folder: str = './test',
        **kwargs,
    ) -> None:
        super().__init__(
            dataset_path=dataset_path,
            name=name,
            cache_dir=cache_dir,
            use_cache=use_cache,
            class_weights=class_weights,
            ignored_label_inds=ignored_label_inds,
            test_result_folder=test_result_folder,
            **kwargs,
        )

        self.label_to_names = self.get_label_to_names()
        self.num_classes = len(self.label_to_names)

    @staticmethod
    def get_label_to_names():
        return {
            0: 'unlabeled',
            1: 'car',
            2: 'bicycle',
            3: 'motorcycle',
            4: 'truck',
            5: 'other-vehicle',
            6: 'person',
            7: 'bicyclist',
            8: 'motorcyclist',
            9: 'road',
            10: 'parking',
            11: 'sidewalk',
            12: 'other-ground',
            13: 'building',
            14: 'fence',
            15: 'vegetation',
            16: 'trunk',
            17: 'terrain',
            18: 'pole',
            19: 'traffic-sign'
        }

    def get_split(self, split):
        return LarkiSplit(self, split=split)

    def get_split_list(self, split):
        if split != "test":
            raise ValueError("Invalid split, only test is supported.")

        return [str(p) for p in Path(self.cfg.dataset_path).iterdir() if p.name.endswith(".e57")]

    def is_tested():
        pass

    def save_test_result():
        pass


class LarkiSplit(BaseDatasetSplit):

    def __init__(self, dataset: BaseDataset, split: str = "test") -> None:
        super().__init__(dataset, split=split)
        logger.info(f"Found {len(self.path_list)} pointclouds for {split}")

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx: int) -> Dict[str, Optional[np.ndarray]]:
        pc_path = self.path_list[idx]

        pcds = e57.load_point_clouds_open3d(pc_path)
        if len(pcds) > 1:
            raise ValueError("Only .e57 files with one scan are supported.")
        points = np.asarray(pcds[0].points)
        points = points - np.max(points, axis=0)

        labels = np.zeros(np.shape(points)[0], dtype=np.int32)
        if self.split != "test":
            raise ValueError("Only test split supported.")

        data = {
            "point": points[:, 0:3],
            "feat": None,
            "label": labels,
        }

        return data

    def get_attr(self, idx):
        pc_path = self.path_list[idx]
        dir, file = os.path.split(pc_path)
        _, seq = os.path.split(os.path.split(dir)[0])
        name = f"{seq}_{file[:-4]}"

        pc_path = str(pc_path)
        attr = {"idx": idx, "name": name, "path": pc_path, "split": self.split}

        return attr


DATASET._register_module(Larki)
