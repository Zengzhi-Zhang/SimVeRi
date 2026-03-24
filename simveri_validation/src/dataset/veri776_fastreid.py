# src/dataset/veri776_fastreid.py
"""
VeRi-776 dataset registration for FastReID.
Follows the same pattern as simveri_fastreid.py.
"""
import os
import random
from src.path_utils import get_default_veri776_root

VERI776_ROOT = get_default_veri776_root()

# Few-shot configuration (set via set_fewshot_config before dataset init)
_FEWSHOT_RATIO = 1.0
_FEWSHOT_SEED = 42


def set_fewshot_config(ratio: float, seed: int = 42):
    """Set the few-shot sampling ratio and seed (must be called before dataset construction)."""
    global _FEWSHOT_RATIO, _FEWSHOT_SEED
    _FEWSHOT_RATIO = ratio
    _FEWSHOT_SEED = seed


def collect_veri776_pids(train_dir: str):
    """Collect all vehicle PIDs from a VeRi-776 training directory.

    Returns a sorted list of integer PIDs.
    """
    pids = set()
    for img_name in os.listdir(train_dir):
        if img_name.endswith('.jpg'):
            parts = img_name.split('_')
            if len(parts) >= 2:
                pids.add(int(parts[0]))
    return sorted(pids)


def sample_fewshot_pids(all_sorted_pids, ratio: float, seed: int):
    """Deterministically sample a subset of PIDs for few-shot training.

    This is the **single source of truth** for few-shot ID selection.
    All code paths (dataset class, count_train_ids, PID saver) must call
    this function to guarantee consistency.

    Returns a sorted list of selected PIDs.
    """
    k = max(1, int(len(all_sorted_pids) * ratio))
    rng = random.Random(seed)
    return sorted(rng.sample(all_sorted_pids, k))


def resolve_veri776_dir(root):
    if root and os.path.isdir(os.path.join(root, "image_train")):
        return root
    return VERI776_ROOT

try:
    from fastreid.data.datasets import DATASET_REGISTRY
    from fastreid.data.datasets.bases import ImageDataset
    FASTREID_AVAILABLE = True
except ImportError:
    FASTREID_AVAILABLE = False
    print("Warning: FastReID not installed.")

if FASTREID_AVAILABLE:

    @DATASET_REGISTRY.register()
    class VeRi776(ImageDataset):
        """VeRi-776 dataset for FastReID.

        Directory layout::
            image_train/   VVVV_cCCC_TTTTTTTT_S.jpg
            image_test/    (gallery)
            image_query/
        """
        dataset_name = "veri776"

        def __init__(self, root='datasets', **kwargs):
            self.root = root
            self.dataset_dir = resolve_veri776_dir(root)

            required_files = [
                os.path.join(self.dataset_dir, 'image_train'),
                os.path.join(self.dataset_dir, 'image_test'),
                os.path.join(self.dataset_dir, 'image_query'),
            ]
            self.check_before_run(required_files)

            # Build vehicle ID -> contiguous index mapping
            self.pid_map = self._build_pid_map()

            train = self._process_dir('image_train')
            query = self._process_dir('image_query')
            gallery = self._process_dir('image_test')

            super().__init__(train, query, gallery, **kwargs)

        def _build_pid_map(self):
            """Build vehicle ID to contiguous index mapping from training set."""
            train_dir = os.path.join(self.dataset_dir, 'image_train')
            sorted_pids = collect_veri776_pids(train_dir)
            pid_map = {pid: idx for idx, pid in enumerate(sorted_pids)}
            print(f"[VeRi776] ID mapping: {len(pid_map)} vehicles "
                  f"(raw ID range: {min(sorted_pids)}-{max(sorted_pids)} -> mapped to 0-{len(pid_map)-1})")
            return pid_map

        def _process_dir(self, dir_name):
            img_dir = os.path.join(self.dataset_dir, dir_name)
            data = []
            for img_name in os.listdir(img_dir):
                if not img_name.endswith('.jpg'):
                    continue
                parts = img_name.replace('.jpg', '').split('_')
                if len(parts) < 2:
                    continue
                raw_pid = int(parts[0])

                if dir_name == 'image_train':
                    pid = self.pid_map[raw_pid]
                else:
                    pid = self.pid_map.get(raw_pid, raw_pid)

                camid = int(parts[1].replace('c', ''))
                img_path = os.path.join(img_dir, img_name)
                data.append((img_path, pid, camid))
            return data


    @DATASET_REGISTRY.register()
    class VeRi776FewShot(ImageDataset):
        """VeRi-776 with reduced training IDs for few-shot experiments.

        Uses module-level _FEWSHOT_RATIO / _FEWSHOT_SEED (set via
        ``set_fewshot_config``) to sample a subset of vehicle IDs for
        training.  Query and Gallery are kept intact.
        """
        dataset_name = "veri776fewshot"

        def __init__(self, root='datasets', **kwargs):
            self.root = root
            self.dataset_dir = resolve_veri776_dir(root)

            required_files = [
                os.path.join(self.dataset_dir, 'image_train'),
                os.path.join(self.dataset_dir, 'image_test'),
                os.path.join(self.dataset_dir, 'image_query'),
            ]
            self.check_before_run(required_files)

            self.selected_pids = None  # populated by _build_pid_map
            self.pid_map = self._build_pid_map()

            train = self._process_dir('image_train')
            query = self._process_dir('image_query')
            gallery = self._process_dir('image_test')

            super().__init__(train, query, gallery, **kwargs)

        def _build_pid_map(self):
            """Build pid_map using only a random subset of training IDs."""
            train_dir = os.path.join(self.dataset_dir, 'image_train')
            sorted_pids = collect_veri776_pids(train_dir)
            selected = sample_fewshot_pids(sorted_pids, _FEWSHOT_RATIO, _FEWSHOT_SEED)
            self.selected_pids = selected

            pid_map = {pid: idx for idx, pid in enumerate(selected)}
            print(f"[VeRi776FewShot] ratio={_FEWSHOT_RATIO}, seed={_FEWSHOT_SEED}: "
                  f"selected {len(selected)}/{len(sorted_pids)} IDs")
            print(f"[VeRi776FewShot] Selected PIDs: {selected}")
            return pid_map

        def _process_dir(self, dir_name):
            img_dir = os.path.join(self.dataset_dir, dir_name)
            data = []
            for img_name in os.listdir(img_dir):
                if not img_name.endswith('.jpg'):
                    continue
                parts = img_name.replace('.jpg', '').split('_')
                if len(parts) < 2:
                    continue
                raw_pid = int(parts[0])

                if dir_name == 'image_train':
                    if raw_pid not in self.pid_map:
                        continue  # skip IDs not in the few-shot subset
                    pid = self.pid_map[raw_pid]
                else:
                    # Query/Gallery: use full VeRi776 pid_map (build on-the-fly
                    # from all training pids) so that evaluation is consistent.
                    pid = raw_pid

                camid = int(parts[1].replace('c', ''))
                img_path = os.path.join(img_dir, img_name)
                data.append((img_path, pid, camid))
            return data


    # Mixed training configuration
    _MIXED_SIMVERI_ROOT = None

    def set_mixed_simveri_root(simveri_root: str):
        """Set SimVeRi root for mixed training (must be called before dataset construction)."""
        global _MIXED_SIMVERI_ROOT
        _MIXED_SIMVERI_ROOT = simveri_root

    @DATASET_REGISTRY.register()
    class VeRi776SimVeRiMixed(ImageDataset):
        """VeRi-776 few-shot + SimVeRi combined training dataset.

        Training set: all SimVeRi images + selected VeRi-776 few-shot images.
        Query/Gallery: VeRi-776 only (unchanged).
        PID mapping: SimVeRi -> 0..Ns-1, VeRi -> Ns..Ns+Nv-1 (disjoint).
        """
        dataset_name = "veri776simverimixed"

        def __init__(self, root='datasets', **kwargs):
            self.root = root
            self.veri_dir = resolve_veri776_dir(root)

            # Resolve SimVeRi root
            from src.dataset.simveri_fastreid import resolve_dataset_dir, SIMVERI_ROOT
            if _MIXED_SIMVERI_ROOT:
                self.simveri_dir = resolve_dataset_dir(_MIXED_SIMVERI_ROOT)
            else:
                self.simveri_dir = resolve_dataset_dir(SIMVERI_ROOT)

            required_files = [
                os.path.join(self.veri_dir, 'image_train'),
                os.path.join(self.veri_dir, 'image_test'),
                os.path.join(self.veri_dir, 'image_query'),
                os.path.join(self.simveri_dir, 'images', 'train'),
            ]
            self.check_before_run(required_files)

            # Build combined PID maps
            self.num_simveri_ids = 0
            self.num_veri_ids = 0
            self.simveri_pid_map, self.veri_pid_map = self._build_pid_maps()

            train = self._build_train()
            query = self._process_veri_dir('image_query')
            gallery = self._process_veri_dir('image_test')

            super().__init__(train, query, gallery, **kwargs)

        def _build_pid_maps(self):
            """Build disjoint PID maps: SimVeRi 0..Ns-1, VeRi Ns..Ns+Nv-1."""
            # SimVeRi PIDs
            sim_train_dir = os.path.join(self.simveri_dir, 'images', 'train')
            sim_pids = set()
            for img_name in os.listdir(sim_train_dir):
                if img_name.endswith('.jpg'):
                    parts = img_name.split('_')
                    if len(parts) >= 2:
                        sim_pids.add(int(parts[0]))
            sim_sorted = sorted(sim_pids)
            sim_pid_map = {pid: idx for idx, pid in enumerate(sim_sorted)}
            self.num_simveri_ids = len(sim_sorted)

            # VeRi few-shot PIDs (offset by Ns)
            veri_train_dir = os.path.join(self.veri_dir, 'image_train')
            veri_all_pids = collect_veri776_pids(veri_train_dir)
            if _FEWSHOT_RATIO < 1.0:
                veri_selected = sample_fewshot_pids(veri_all_pids, _FEWSHOT_RATIO, _FEWSHOT_SEED)
            else:
                veri_selected = veri_all_pids
            offset = self.num_simveri_ids
            veri_pid_map = {pid: idx + offset for idx, pid in enumerate(veri_selected)}
            self.num_veri_ids = len(veri_selected)

            total = self.num_simveri_ids + self.num_veri_ids
            print(f"[VeRi776SimVeRiMixed] SimVeRi: {self.num_simveri_ids} IDs (0..{self.num_simveri_ids-1}), "
                  f"VeRi: {self.num_veri_ids} IDs ({offset}..{total-1}), "
                  f"Total: {total} IDs")
            return sim_pid_map, veri_pid_map

        def _build_train(self):
            """Concatenate SimVeRi + VeRi few-shot training images."""
            data = []

            # SimVeRi training images (camid offset by 100 to avoid collision)
            sim_train_dir = os.path.join(self.simveri_dir, 'images', 'train')
            for img_name in os.listdir(sim_train_dir):
                if not img_name.endswith('.jpg'):
                    continue
                parts = img_name.replace('.jpg', '').split('_')
                if len(parts) < 2:
                    continue
                raw_pid = int(parts[0])
                if raw_pid not in self.simveri_pid_map:
                    continue
                pid = self.simveri_pid_map[raw_pid]
                camid = int(parts[1].replace('c', '')) + 100
                img_path = os.path.join(sim_train_dir, img_name)
                data.append((img_path, pid, camid))

            sim_count = len(data)

            # VeRi few-shot training images
            veri_train_dir = os.path.join(self.veri_dir, 'image_train')
            for img_name in os.listdir(veri_train_dir):
                if not img_name.endswith('.jpg'):
                    continue
                parts = img_name.replace('.jpg', '').split('_')
                if len(parts) < 2:
                    continue
                raw_pid = int(parts[0])
                if raw_pid not in self.veri_pid_map:
                    continue
                pid = self.veri_pid_map[raw_pid]
                camid = int(parts[1].replace('c', ''))
                img_path = os.path.join(veri_train_dir, img_name)
                data.append((img_path, pid, camid))

            veri_count = len(data) - sim_count
            print(f"[VeRi776SimVeRiMixed] Train images: {sim_count} SimVeRi + {veri_count} VeRi = {len(data)} total")
            return data

        def _process_veri_dir(self, dir_name):
            """Process VeRi-776 query/gallery (raw PIDs, same as VeRi776FewShot)."""
            img_dir = os.path.join(self.veri_dir, dir_name)
            data = []
            for img_name in os.listdir(img_dir):
                if not img_name.endswith('.jpg'):
                    continue
                parts = img_name.replace('.jpg', '').split('_')
                if len(parts) < 2:
                    continue
                raw_pid = int(parts[0])
                camid = int(parts[1].replace('c', ''))
                img_path = os.path.join(img_dir, img_name)
                data.append((img_path, raw_pid, camid))
            return data


if __name__ == "__main__":
    if not FASTREID_AVAILABLE:
        print("FastReID not available, skipping tests.")
    else:
        print("Testing VeRi-776 FastReID dataset registration...")
        ds = VeRi776(root=VERI776_ROOT)
        print(f"Train: {len(ds.train)}, Query: {len(ds.query)}, Gallery: {len(ds.gallery)}")
        print("Done!")
