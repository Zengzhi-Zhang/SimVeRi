# src/models/feature_extractor.py
"""
Feature extractor - uses FastReID-style preprocessing for consistency
"""

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Optional, Union
import cv2


class FeatureExtractor:
    """Feature extractor."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = 'cuda',
                 input_size: Union[int, tuple] = 256):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_fastreid = False
        self.model = None
        self.transform = None

        # Keep a consistent HxW convention (same as FastReID cfg.INPUT.SIZE_*).
        if isinstance(input_size, int):
            self.input_size = (int(input_size), int(input_size))  # (H, W)
        elif isinstance(input_size, (tuple, list)) and len(input_size) == 2:
            self.input_size = (int(input_size[0]), int(input_size[1]))  # (H, W)
        else:
            self.input_size = (256, 256)
        
        print(f"Initializing FeatureExtractor...")
        print(f"  Device: {self.device}")
        print(f"  Input size (HxW): {self.input_size[0]}x{self.input_size[1]}")
        
        if model_path and os.path.exists(model_path):
            try:
                self._build_fastreid_model(model_path)
                self.use_fastreid = True
            except Exception as e:
                print(f"  Failed to build FastReID model: {e}")
                import traceback
                traceback.print_exc()
                print(f"  Falling back to torchvision model")
                self._setup_torchvision()
        else:
            print(f"  No model provided, using torchvision ResNet-50")
            self._setup_torchvision()
        
        print(f"  Model loaded successfully")

    def _infer_num_classes_from_checkpoint(self, model_path: str) -> Optional[int]:
        """Infer classifier size from checkpoint when available."""
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
        except Exception as e:
            print(f"  Warning: Failed to read checkpoint metadata: {e}")
            return None

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            return None

        preferred_keys = [
            "heads.classifier.weight",
            "heads.weight",
            "classifier.weight",
            "module.heads.classifier.weight",
            "module.heads.weight",
            "module.classifier.weight",
        ]
        for key in preferred_keys:
            if key in state_dict:
                try:
                    return int(state_dict[key].shape[0])
                except Exception:
                    return None

        for key, value in state_dict.items():
            if key.endswith("heads.classifier.weight") or key.endswith("heads.weight") or key.endswith("classifier.weight"):
                try:
                    return int(value.shape[0])
                except Exception:
                    return None

        return None

    def _build_fastreid_model(self, model_path: str):
        """Build a FastReID model directly without DefaultPredictor."""
        from fastreid.config import get_cfg
        from fastreid.modeling import build_model
        from fastreid.utils.checkpoint import Checkpointer
        
        cfg = get_cfg()
        
        is_veri_pretrained = 'veri_sbs' in model_path.lower() or 'veri-776' in model_path.lower()
        
        cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
        cfg.MODEL.BACKBONE.DEPTH = "50x"
        cfg.MODEL.BACKBONE.WITH_IBN = True
        cfg.MODEL.BACKBONE.PRETRAIN = False
        
        cfg.MODEL.HEADS.NAME = "EmbeddingHead"
        cfg.MODEL.HEADS.POOL_LAYER = "GlobalAvgPool"
        cfg.MODEL.HEADS.NECK_FEAT = "after"
        cfg.MODEL.HEADS.WITH_BNNECK = True
        
        inferred_classes = self._infer_num_classes_from_checkpoint(model_path)
        if inferred_classes:
            cfg.MODEL.HEADS.NUM_CLASSES = inferred_classes
            print(f"  Model classes: {inferred_classes} (from checkpoint)")
        elif is_veri_pretrained:
            cfg.MODEL.HEADS.NUM_CLASSES = 576
            print(f"  Model type: VeRi pretrained")
        else:
            cfg.MODEL.HEADS.NUM_CLASSES = 530
            print(f"  Model type: SimVeRi trained")
        
        cfg.MODEL.DEVICE = str(self.device)
        
        self.model = build_model(cfg)
        self.model.eval()
        self.model.to(self.device)
        
        Checkpointer(self.model).load(model_path)
        
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(self.device)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(self.device)
        # NOTE: self.input_size is set by ctor (HxW). Keep it to align with training/eval.
        
        print(f"  Loading model: {model_path}")
        print(f"  Normalization: Caffe-style (0-255 range)")
    
    def _setup_torchvision(self):
        """Set up the torchvision model."""
        import torchvision.models as models
        import torchvision.transforms as T
        import torch.nn as nn
        
        resnet = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.model.eval()
        self.model.to(self.device)
        
        self.transform = T.Compose([
            T.Resize(self.input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
        
        print(f"  Loaded torchvision ResNet-50 (ImageNet pretrained)")
    
    def _preprocess_fastreid(self, img: Image.Image) -> torch.Tensor:
        """Apply FastReID-style preprocessing."""
        # Resize
        h, w = self.input_size
        img = img.resize((w, h))
        
        # PIL -> numpy (RGB, uint8 in [0, 255])
        # Keep RGB order (FastReID transforms are RGB-based) and keep raw 0-255 range.
        img_np = np.array(img).copy()
        
        # HWC -> CHW, numpy -> tensor
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
        
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # NOTE: Do NOT normalize here. FastReID Baseline.preprocess_image() already applies
        # (img - pixel_mean) / pixel_std once.
        return img_tensor
    
    @torch.no_grad()
    def extract_single(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """Extract features for one image."""
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert('RGB')
        else:
            img = image.convert('RGB')
        
        if self.use_fastreid:
            img_tensor = self._preprocess_fastreid(img)
            features = self.model(img_tensor)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            return features.cpu().numpy().flatten()
        else:
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            features = self.model(img_tensor)
            features = features.squeeze(-1).squeeze(-1)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            return features.cpu().numpy().flatten()
    
    @torch.no_grad()
    def extract_batch(self, 
                      image_paths: List[str], 
                      batch_size: int = 64,
                      show_progress: bool = True) -> np.ndarray:
        """Extract features in batches."""
        all_features = []
        
        num_batches = (len(image_paths) + batch_size - 1) // batch_size
        iterator = range(0, len(image_paths), batch_size)
        
        if show_progress:
            iterator = tqdm(iterator, total=num_batches, desc="Extracting features")
        
        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, len(image_paths))
            batch_paths = image_paths[start_idx:end_idx]
            
            batch_features = []
            for path in batch_paths:
                try:
                    feat = self.extract_single(path)
                    batch_features.append(feat)
                except Exception as e:
                    print(f"\n  Warning: Failed to process {path}: {e}")
                    batch_features.append(np.zeros(2048))
            
            all_features.extend(batch_features)
        
        features = np.array(all_features)
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
        features = features / norms
        
        return features
    
    def get_feature_dim(self) -> int:
        """Return the feature dimensionality."""
        return 2048
