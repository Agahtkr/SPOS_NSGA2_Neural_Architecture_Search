# data/dataloader.py
import os
import cv2
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DetectionDataset(Dataset):
    def __init__(self, image_paths, label_paths, config, is_train=True):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.config = config
        self.is_train = is_train
        self.transform = self._build_transforms()

    def _build_transforms(self):
        if not self.is_train:
            return A.Compose([
                A.Resize(self.config.input_resolution, self.config.input_resolution),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        train_transforms = []

        # FIX: HueSaturationValue converts BGR<->HSV internally and requires
        # a 3-channel color image. With config.in_channels == 1 (grayscale),
        # cv2.imread(..., IMREAD_GRAYSCALE) produces a (H, W, 1) array, and
        # this transform raises a cv2 error the moment training starts.
        # Only include color-space augmentations when actually working with
        # color images.
        if self.config.in_channels == 3:
            train_transforms.append(
                A.HueSaturationValue(hue_shift_limit=1, sat_shift_limit=127, val_shift_limit=102, p=0.5)
            )

        train_transforms.extend([
            A.ShiftScaleRotate(
                shift_limit=0.10, scale_limit=0.40, rotate_limit=5.0,
                border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114), p=0.5
            ),
            A.HorizontalFlip(p=0.5),
            A.Resize(self.config.input_resolution, self.config.input_resolution),
            ToTensorV2()
        ])

        return A.Compose(
            train_transforms,
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], clip=True, min_visibility=0.2)
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        if self.config.in_channels == 1:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image.ndim == 2:
                image = image[..., None]
        else:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        lbl_path = self.label_paths[idx]
        bboxes, class_labels = [], []

        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        old_class_id = int(parts[0])

                        if hasattr(self.config, 'class_mapping') and self.config.class_mapping:
                            mapped_id = self.config.class_mapping.get(old_class_id)
                            if mapped_id is not None:
                                class_labels.append(mapped_id)
                                bboxes.append([float(x) for x in parts[1:]])
                        else:
                            class_labels.append(old_class_id)
                            bboxes.append([float(x) for x in parts[1:]])

        # --- DİKKAT: Burası 'if' bloğunun DIŞINDA olmalı ---
        safe_bboxes = []
        safe_labels = []

        for box, label in zip(bboxes, class_labels):
            cx, cy, w, h = box

            x_min = cx - w / 2.0
            y_min = cy - h / 2.0
            x_max = cx + w / 2.0
            y_max = cy + h / 2.0

            # Float hatasını (negatife düşmeyi) engellemek için 0.0 yerine 0.00001 kullanıyoruz
            x_min = max(0.00001, x_min)
            y_min = max(0.00001, y_min)
            x_max = min(0.99999, x_max)
            y_max = min(0.99999, y_max)

            new_w = x_max - x_min
            new_h = y_max - y_min

            if new_w > 0.001 and new_h > 0.001:
                new_cx = x_min + new_w / 2.0
                new_cy = y_min + new_h / 2.0
                # Yuvarlama yapmadan, ham ve güvenli koordinatları gönderiyoruz
                safe_bboxes.append([new_cx, new_cy, new_w, new_h])
                safe_labels.append(label)

        bboxes = safe_bboxes
        class_labels = safe_labels

        # Bu dönüşüm işlemi de 'if' bloğunun dışında olmalı
        transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)

        image_tensor = transformed['image'].float() / 255.0
        bbox_tensor = torch.tensor(transformed['bboxes'], dtype=torch.float32)
        class_tensor = torch.tensor(transformed['class_labels'], dtype=torch.int64)

        return image_tensor, bbox_tensor, class_tensor

def yolo_collate_fn(batch):
    images, targets = [], []
    for i, (img, box, cls) in enumerate(batch):
        images.append(img)
        if len(box) > 0:
            batch_idx = torch.full((box.shape[0], 1), i, dtype=torch.float32)
            targets.append(torch.cat((batch_idx, cls.unsqueeze(1).float(), box), dim=1))

    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0) if len(targets) > 0 else torch.empty((0, 6))
    return images, targets


class DataManager:
    def __init__(self, config):
        self.config = config

    def _get_image_label_pairs(self, img_dir, lbl_dir):
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
            image_paths.extend(glob.glob(os.path.join(img_dir, ext)))

        image_paths.sort()
        label_paths = [
            os.path.join(lbl_dir, f"{os.path.splitext(os.path.basename(p))[0]}.txt")
            for p in image_paths
        ]
        return image_paths, label_paths

    def get_dataloaders(self, train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir):
        train_imgs, train_lbls = self._get_image_label_pairs(train_img_dir, train_lbl_dir)
        val_imgs, val_lbls = self._get_image_label_pairs(val_img_dir, val_lbl_dir)

        train_loader = DataLoader(
            DetectionDataset(train_imgs, train_lbls, self.config, is_train=True),
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=yolo_collate_fn,
            num_workers=4,
            drop_last=True,
            pin_memory=True,  # Kilitli bellek (Hızlı transfer)
            persistent_workers=True  # Worker'ları hayatta tut
        )
        val_loader = DataLoader(
            DetectionDataset(val_imgs, val_lbls, self.config, is_train=False),
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=yolo_collate_fn,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        return train_loader, val_loader