# src/dataset/simveri_fastreid.py
import os
from src.path_utils import get_default_simveri_root

SIMVERI_ROOT = get_default_simveri_root()


def resolve_dataset_dir(root):
    if root and os.path.isdir(os.path.join(root, "images", "train")):
        return root
    return SIMVERI_ROOT

try:
    from fastreid.data.datasets import DATASET_REGISTRY
    from fastreid.data.datasets.bases import ImageDataset
    FASTREID_AVAILABLE = True
except ImportError:
    FASTREID_AVAILABLE = False
    print("Warning: FastReID not installed.")

if FASTREID_AVAILABLE:

    @DATASET_REGISTRY.register()
    class SimVeRi(ImageDataset):
        dataset_name = "simveri"

        def __init__(self, root='datasets', **kwargs):
            self.root = root
            self.dataset_dir = resolve_dataset_dir(root)

            required_files = [
                os.path.join(self.dataset_dir, 'images', 'train'),
                os.path.join(self.dataset_dir, 'images', 'gallery'),
                os.path.join(self.dataset_dir, 'images', 'query'),
            ]
            self.check_before_run(required_files)

            self.pid_map = self._build_pid_map()

            train = self._process_dir('train')
            query = self._process_dir('query')
            gallery = self._process_dir('gallery')

            super().__init__(train, query, gallery, **kwargs)

        def _build_pid_map(self):
            """Build a mapping from vehicle IDs to contiguous indices."""
            train_dir = os.path.join(self.dataset_dir, 'images', 'train')
            pids = set()
            for img_name in os.listdir(train_dir):
                if img_name.endswith('.jpg'):
                    parts = img_name.split('_')
                    if len(parts) >= 2:
                        pids.add(int(parts[0]))
            pid_map = {pid: idx for idx, pid in enumerate(sorted(pids))}
            print(f"[SimVeRi] ID mapping: {len(pid_map)} vehicles (raw range: {min(pids)}-{max(pids)} -> mapped to 0-{len(pid_map)-1})")
            return pid_map

        def _process_dir(self, split):
            img_dir = os.path.join(self.dataset_dir, 'images', split)
            data = []
            for img_name in os.listdir(img_dir):
                if not img_name.endswith('.jpg'):
                    continue
                parts = img_name.replace('.jpg', '').split('_')
                if len(parts) < 2:
                    continue
                raw_pid = int(parts[0])

                if split == 'train':
                    pid = self.pid_map[raw_pid]
                else:
                    pid = self.pid_map.get(raw_pid, raw_pid)

                camid = int(parts[1].replace('c', ''))
                img_path = os.path.join(img_dir, img_name)
                data.append((img_path, pid, camid))
            return data
    
    
    @DATASET_REGISTRY.register()
    class SimVeRiBase(ImageDataset):
        dataset_name = "simveri_base"
        TWINS_ID_START = 431
        TWINS_ID_END = 530
        
        def __init__(self, root='datasets', **kwargs):
            self.root = root
            self.dataset_dir = resolve_dataset_dir(root)

            # Build vehicle ID -> contiguous index mapping from the (filtered) training set.
            self.pid_map = self._build_pid_map(exclude_twins=True)
            
            train = self._process_dir('train', exclude_twins=True)
            query = self._process_dir('query', exclude_twins=True)
            gallery = self._process_dir('gallery', exclude_twins=True)
            
            super().__init__(train, query, gallery, **kwargs)

        def _build_pid_map(self, exclude_twins=False):
            train_dir = os.path.join(self.dataset_dir, 'images', 'train')
            pids = set()
            for img_name in os.listdir(train_dir):
                if not img_name.endswith('.jpg'):
                    continue
                parts = img_name.split('_')
                if len(parts) < 2:
                    continue
                pid = int(parts[0])
                if exclude_twins and self._is_twins(pid):
                    continue
                pids.add(pid)

            pid_map = {pid: idx for idx, pid in enumerate(sorted(pids))}
            if pids:
                print(f"[SimVeRiBase] ID mapping: {len(pid_map)} vehicles "
                      f"(raw ID range: {min(pids)}-{max(pids)} -> mapped to 0-{len(pid_map)-1})")
            else:
                print("[SimVeRiBase] ID mapping: 0 vehicles (empty train setgroups)")
            return pid_map
        
        def _is_twins(self, pid):
            return self.TWINS_ID_START <= pid <= self.TWINS_ID_END
        
        def _process_dir(self, split, exclude_twins=False):
            img_dir = os.path.join(self.dataset_dir, 'images', split)
            data = []
            for img_name in os.listdir(img_dir):
                if not img_name.endswith('.jpg'):
                    continue
                parts = img_name.replace('.jpg', '').split('_')
                if len(parts) < 2:
                    continue
                raw_pid = int(parts[0])
                camid = int(parts[1].replace('c', ''))
                if exclude_twins and self._is_twins(raw_pid):
                    continue

                if split == 'train':
                    pid = self.pid_map[raw_pid]
                else:
                    pid = self.pid_map.get(raw_pid, raw_pid)
                img_path = os.path.join(img_dir, img_name)
                data.append((img_path, pid, camid))
            return data
    
    
    @DATASET_REGISTRY.register()
    class SimVeRiTwins(ImageDataset):
        dataset_name = "simveri_twins"
        TWINS_ID_START = 431
        TWINS_ID_END = 530
        
        def __init__(self, root='datasets', **kwargs):
            self.root = root
            self.dataset_dir = resolve_dataset_dir(root)

            # Build vehicle ID -> contiguous index mapping from the (filtered) training set.
            self.pid_map = self._build_pid_map(twins_only=True)
            
            train = self._process_dir('train', twins_only=True)
            query = self._process_dir('query', twins_only=True)
            gallery = self._process_dir('gallery', twins_only=True)
            
            super().__init__(train, query, gallery, **kwargs)

        def _build_pid_map(self, twins_only=False):
            train_dir = os.path.join(self.dataset_dir, 'images', 'train')
            pids = set()
            for img_name in os.listdir(train_dir):
                if not img_name.endswith('.jpg'):
                    continue
                parts = img_name.split('_')
                if len(parts) < 2:
                    continue
                pid = int(parts[0])
                if twins_only and not self._is_twins(pid):
                    continue
                pids.add(pid)

            pid_map = {pid: idx for idx, pid in enumerate(sorted(pids))}
            if pids:
                print(f"[SimVeRiTwins] ID mapping: {len(pid_map)} vehicles "
                      f"(raw ID range: {min(pids)}-{max(pids)} -> mapped to 0-{len(pid_map)-1})")
            else:
                print("[SimVeRiTwins] ID mapping: 0 vehicles (empty train setgroups)")
            return pid_map
        
        def _is_twins(self, pid):
            return self.TWINS_ID_START <= pid <= self.TWINS_ID_END
        
        def _process_dir(self, split, twins_only=False):
            img_dir = os.path.join(self.dataset_dir, 'images', split)
            data = []
            for img_name in os.listdir(img_dir):
                if not img_name.endswith('.jpg'):
                    continue
                parts = img_name.replace('.jpg', '').split('_')
                if len(parts) < 2:
                    continue
                raw_pid = int(parts[0])
                camid = int(parts[1].replace('c', ''))
                if twins_only and not self._is_twins(raw_pid):
                    continue

                if split == 'train':
                    pid = self.pid_map[raw_pid]
                else:
                    pid = self.pid_map.get(raw_pid, raw_pid)
                img_path = os.path.join(img_dir, img_name)
                data.append((img_path, pid, camid))
            return data


if __name__ == "__main__":
    if not FASTREID_AVAILABLE:
        print("FastReID not available, skipping tests.")
    else:
        print("Testing FastReID dataset registration...")
        
        root = get_default_simveri_root()
        
        print("\n--- SimVeRi (Full) ---")
        ds = SimVeRi(root=root)
        print(f"Train: {len(ds.train)}, Query: {len(ds.query)}, Gallery: {len(ds.gallery)}")
        
        print("\n--- SimVeRiBase ---")
        ds_base = SimVeRiBase(root=root)
        print(f"Train: {len(ds_base.train)}, Query: {len(ds_base.query)}, Gallery: {len(ds_base.gallery)}")
        
        print("\n--- SimVeRiTwins ---")
        ds_twins = SimVeRiTwins(root=root)
        print(f"Train: {len(ds_twins.train)}, Query: {len(ds_twins.query)}, Gallery: {len(ds_twins.gallery)}")
        
        print("\n[OK] All FastReID registrations successful")
