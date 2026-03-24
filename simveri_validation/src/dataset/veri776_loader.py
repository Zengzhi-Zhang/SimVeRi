# src/dataset/veri776_loader.py
"""
VeRi-776 dataset loader for feature extraction.
Simplified version of simveri_loader.py - no XML annotations or spatiotemporal metadata.
File name convention: VVVV_cCCC_TTTTTTTT_S.jpg  (same as SimVeRi)
"""

import os
from dataclasses import dataclass
from typing import Dict, List

from src.path_utils import get_default_veri776_root


@dataclass
class VeRi776Sample:
    """Single VeRi-776 sample."""
    image_name: str
    image_path: str
    vehicle_id: str
    camera_id: str
    is_twins: bool = False


class VeRi776Dataset:
    """
    VeRi-776 dataset loader for feature extraction.

    Usage:
        dataset = VeRi776Dataset("../external_datasets/VeRi-776/VeRi")
        gallery = dataset.gallery_samples
        query   = dataset.query_samples
    """

    def __init__(self, root_dir: str, verbose: bool = True):
        self.root_dir = root_dir
        self.verbose = verbose

        self.gallery_samples: List[VeRi776Sample] = []
        self.query_samples: List[VeRi776Sample] = []

        self._load(root_dir)

        if verbose:
            self._print_summary()

    def _load(self, root_dir: str):
        test_dir = os.path.join(root_dir, 'image_test')
        query_dir = os.path.join(root_dir, 'image_query')

        if not os.path.isdir(test_dir):
            raise FileNotFoundError(f"image_test directory not found: {test_dir}")
        if not os.path.isdir(query_dir):
            raise FileNotFoundError(f"image_query directory not found: {query_dir}")

        self.gallery_samples = self._parse_dir(test_dir)
        self.query_samples = self._parse_dir(query_dir)

    @staticmethod
    def _parse_dir(img_dir: str) -> List[VeRi776Sample]:
        samples = []
        for img_name in sorted(os.listdir(img_dir)):
            if not img_name.endswith('.jpg'):
                continue
            parts = img_name.replace('.jpg', '').split('_')
            if len(parts) < 2:
                continue
            vid = parts[0]
            cam_part = parts[1]
            cam = cam_part if cam_part.startswith('c') else f"c{cam_part}"

            samples.append(VeRi776Sample(
                image_name=img_name,
                image_path=os.path.join(img_dir, img_name),
                vehicle_id=vid,
                camera_id=cam,
                is_twins=False,
            ))
        return samples

    def _print_summary(self):
        g_vids = {s.vehicle_id for s in self.gallery_samples}
        q_vids = {s.vehicle_id for s in self.query_samples}
        print("=" * 60)
        print("VeRi-776 Dataset Summary")
        print("=" * 60)
        print(f"  Gallery samples: {len(self.gallery_samples):,}  ({len(g_vids)} IDs)")
        print(f"  Query samples:   {len(self.query_samples):,}  ({len(q_vids)} IDs)")
        print("=" * 60)


if __name__ == "__main__":
    root = get_default_veri776_root()
    ds = VeRi776Dataset(root)
