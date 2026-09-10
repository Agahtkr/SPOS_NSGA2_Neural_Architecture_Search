# train_supernet.py
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from configs.search_config import load_config
from data.dataloader import DataManager
from models.supernet import SPOSSupernet
from training.engine import NASEngine


def main():
    cudnn.benchmark = False  # op shapes change every step; benchmark would thrash
    config = load_config()  # configs/config.yaml

    # --- DOSYA YOLLARINI KESİN OLARAK KOD İÇİNDE SABİTLİYORUZ ---
    config.train_img_dir = "/home/aisoft/Aisoft/Gumus-su/0.5L-area/data/dataset_13_08_v2/images/train"
    config.train_lbl_dir = "/home/aisoft/Aisoft/Gumus-su/0.5L-area/data/dataset_13_08_v2/labels/train"
    config.val_image_dir = "/home/aisoft/Aisoft/Gumus-su/0.5L-area/data/dataset_13_08_v2/images/val"
    config.val_lbl_dir = "/home/aisoft/Aisoft/Gumus-su/0.5L-area/data/dataset_13_08_v2/labels/val"

    # Yeni modelin kaydedileceği klasörü de mutlak yol yapıyoruz
    config.save_dir = "/home/aisoft/Aisoft/Gumus-su/0.5L-area/script/nas/experiments"

    # DOSYA YOLU KORUMA KALKANI
    if not os.path.exists(config.train_img_dir):
        raise FileNotFoundError(f"\n[HATA] Veri yolu bulunamadı: {config.train_img_dir}")
    # -------------------------------------------------------------

    print("Initializing Data & Supernet...")
    train_loader, val_loader = DataManager(config).get_dataloaders(
        train_img_dir=config.train_img_dir, train_lbl_dir=config.train_lbl_dir,
        val_img_dir=config.val_image_dir, val_lbl_dir=config.val_lbl_dir,
    )

    supernet = SPOSSupernet(config)
    if torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()} GPUs detected. Enabling DataParallel.")
        supernet = nn.DataParallel(supernet)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = NASEngine(supernet, train_loader, val_loader, config, device=device, enable_logging=True)

    # Süpernet yolunu da config.save_dir üzerinden alacak şekilde bıraktık
    supernet_path = "/home/aisoft/Aisoft/Gumus-su/0.5L-area/script/nas/experiments/supernet_last.pth"

    if os.path.exists(supernet_path):
        start_epoch = engine.load_checkpoint(supernet_path)
        print(f"Kaldığı yerden devam ediliyor: epoch {start_epoch + 1} "
              f"({engine.paths_per_batch} path(s) per batch, {engine.num_ops} ops/layer)...")
    else:
        start_epoch = 0
        print(f"Sıfırdan SPOS eğitimi başlıyor "
              f"({engine.paths_per_batch} path(s) per batch, {engine.num_ops} ops/layer)...")

    os.makedirs(config.save_dir, exist_ok=True)

    try:
        for epoch in range(start_epoch, config.supernet_epochs):
            engine.train_supernet_epoch(epoch)
            engine.save_checkpoint(epoch)
    finally:
        engine.close()  # flush/close TensorBoard writer even on Ctrl+C

    print("Supernet training complete; checkpoint saved to", config.save_dir)


if __name__ == "__main__":
    main()