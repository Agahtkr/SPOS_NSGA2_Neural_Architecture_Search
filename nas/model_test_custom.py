import os
import glob
import torch
import numpy as np
from sklearn.metrics import f1_score, classification_report
from configs.search_config import load_config
from data.dataloader import DetectionDataset, yolo_collate_fn
from torch.utils.data import DataLoader
from models.supernet import StandaloneSubnet


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hem integer hem string formatını ekleyerek olası bir key uyuşmazlığını (KeyError) önlüyoruz
    config.class_mapping = {0: 1, 1: 0, 2: 0, 3: 2, 4: 3, 5: 4,
                            '0': 1, '1': 1, '2': 0, '3': 2, '4': 3, '5': 4}

    # Doğrudan orijinal veri setinin yolları
    TEST_IMG_DIR = "/opt/project/ssd_etiketlenmiş_veri/images"
    TEST_LBL_DIR = "/opt/project/ssd_etiketlenmiş_veri/labels"


    # Senin subnet_best.pth dosyanın yolunu buraya yazdım
    checkpoint_path = "/opt/project/subnet_best.pth"

    best_chromosome = [0, 1, 3, 9, 2, 5, 2, 12, 10, 0, 0, 8]
    model = StandaloneSubnet(config, best_chromosome).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model ağırlığı bulunamadı: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in checkpoint['model_state_dict'].items()}
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
        image_paths.extend(glob.glob(os.path.join(TEST_IMG_DIR, ext)))
    image_paths.sort()

    label_paths = [
        os.path.join(TEST_LBL_DIR, f"{os.path.splitext(os.path.basename(p))[0]}.txt")
        for p in image_paths
    ]

    print(f"Bulunan Görsel: {len(image_paths)} | Beklenen Etiket: {len(label_paths)}")

    val_dataset = DetectionDataset(image_paths, label_paths, config, is_train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=yolo_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    class_names = ['hatali-kapak', 'kapakli', 'kapaksiz', 'sivi-var', 'sivi-yok']
    all_true = []
    all_pred = []

    print("Çifte haritalama engellendi. Test başlatılıyor...")

    toplam_islenen_hedef = 0

    with torch.no_grad():
        for batch_idx_num, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)

            if len(targets) == 0:
                continue

            toplam_islenen_hedef += len(targets)

            batch_idx = targets[:, 0].long()
            class_id = targets[:, 1].long()
            boxes = targets[:, 2:6]
            box_areas = boxes[:, 2] * boxes[:, 3]

            for scale_idx, pred in enumerate(logits):
                B, _, H, W = pred.shape
                pred_logits = pred[:, 4:, :, :]

                if scale_idx == 0:
                    scale_mask = box_areas < 0.05
                elif scale_idx == 1:
                    scale_mask = (box_areas >= 0.05) & (box_areas < 0.2)
                else:
                    scale_mask = box_areas >= 0.2

                if scale_mask.sum() > 0:
                    b_idx_s = batch_idx[scale_mask]
                    c_id_s = class_id[scale_mask]
                    boxes_s = boxes[scale_mask]

                    grid_x = (boxes_s[:, 0] * W).long().clamp(0, W - 1)
                    grid_y = (boxes_s[:, 1] * H).long().clamp(0, H - 1)

                    target_pixel_preds = pred_logits[b_idx_s, :, grid_y, grid_x]
                    predicted_classes = torch.argmax(target_pixel_preds, dim=1)

                    all_true.extend(c_id_s.cpu().numpy())
                    all_pred.extend(predicted_classes.cpu().numpy())

    print(f"DataLoader'dan geçen toplam geçerli etiket (hedef) sayısı: {toplam_islenen_hedef}")

    if len(all_true) > 0:
        all_true = np.array(all_true)
        all_pred = np.array(all_pred)

        macro_f1 = f1_score(all_true, all_pred, average='macro', zero_division=0)
        print("=" * 60)
        print(f"Gerçek Sınıf Dağılımıyla NAS F1 Skoru: {macro_f1:.4f}")
        print("=" * 60)

        present_classes = sorted(list(np.unique(all_true)))
        valid_target_names = [class_names[i] for i in present_classes if i < len(class_names)]

        print(classification_report(all_true, all_pred, labels=present_classes, target_names=valid_target_names,
                                    zero_division=0))
    else:
        print(
            "Veri setinde eşleşen hedef bulunamadı. Muhtemel sebep: scale_mask (kutu boyutları) filtrelemesine takılıyorlar.")


if __name__ == "__main__":
    main()