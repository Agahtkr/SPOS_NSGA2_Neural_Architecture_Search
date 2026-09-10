# run_search.py
import os
import matplotlib
matplotlib.use('Agg')
import json
import random
import torch
from torch.utils.data import DataLoader
from configs.search_config import load_config
from data.dataloader import DataManager, DetectionDataset, yolo_collate_fn
from models.supernet import SPOSSupernet, StandaloneSubnet
from training.engine import NASEngine
from utils.lut import LUTManager
from search.nsga2 import NSGA2


def main():
    config = load_config()  # configs/config.yaml
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Preparing data & supernet for search...")
    data_manager = DataManager(config)
    # --- DOSYA YOLLARINI KESİN OLARAK KOD İÇİNDE SABİTLİYORUZ ---
    config.train_img_dir = "/home/aisoft/Aisoft/Gumus-su/0.5L-area/data/dataset_13_08_v2/images/train"
    config.train_lbl_dir = "/home/aisoft/Aisoft/Gumus-su/0.5L-area/data/dataset_13_08_v2/labels/train"
    config.val_image_dir = "/home/aisoft/Aisoft/Gumus-su/0.5L-area/data/dataset_13_08_v2/images/val"
    config.val_lbl_dir = "/home/aisoft/Aisoft/Gumus-su/0.5L-area/data/dataset_13_08_v2/labels/val"
    _, val_loader = data_manager.get_dataloaders(
        train_img_dir=config.train_img_dir, train_lbl_dir=config.train_lbl_dir,
        val_img_dir=config.val_image_dir, val_lbl_dir=config.val_lbl_dir,
    )

    # BN-calibration subset: a fixed random sample (sorted[:N] would be biased
    # toward whichever class sorts first alphabetically).
    train_imgs, train_lbls = data_manager._get_image_label_pairs(config.train_img_dir, config.train_lbl_dir)
    calib_size = min(len(train_imgs), config.bn_calib_batches * config.batch_size)
    rng = random.Random(0)
    calib_idx = rng.sample(range(len(train_imgs)), calib_size)
    calib_dataset = DetectionDataset([train_imgs[i] for i in calib_idx], [train_lbls[i] for i in calib_idx],
                                     config, is_train=False)
    calib_loader = DataLoader(calib_dataset, batch_size=config.batch_size, shuffle=False,
                              num_workers=2, collate_fn=yolo_collate_fn, pin_memory=(device == "cuda"))

    supernet = SPOSSupernet(config)
    weights_path = os.path.join("/home/aisoft/Aisoft/Gumus-su/0.5L-area/script/nas/experiments/supernet_last.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"No supernet weights at {weights_path}. Run train_supernet.py first.")

    checkpoint = torch.load(weights_path, map_location="cpu")
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    supernet.load_state_dict(state)
    print(f"=> Loaded supernet weights (epoch: {checkpoint.get('epoch', 'unknown') if isinstance(checkpoint, dict) else 'raw'}).")

    # Single device on purpose: BN running-stat updates are unreliable under DataParallel.
    engine = NASEngine(supernet, calib_loader, val_loader, config, device=device, enable_logging=False)
    lut = LUTManager(config, device="cpu")

    final_fronts, final_pop, final_fitness = NSGA2(config, engine, lut).run()

    pareto_architectures = [final_pop[i] for i in final_fronts[0]]
    pareto_metrics = [final_fitness[i] for i in final_fronts[0]]

    print("\n--- Search complete ---")
    best_models_dir = os.path.join(config.save_dir, "best_models")
    os.makedirs(best_models_dir, exist_ok=True)

    for idx, (arch, (f1, lat)) in enumerate(zip(pareto_architectures, pareto_metrics)):
        print(f"Model {idx + 1} | Architecture: {arch} | F1: {f1:.4f} | Latency: {lat:.2f}ms")

        model = StandaloneSubnet(config, subnet_config=arch, supernet=engine.model).to(device)

        model.train()
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.reset_running_stats()
                m.momentum = None
        with torch.no_grad():
            for i, (x, _) in enumerate(calib_loader):
                if i >= config.bn_calib_batches:
                    break
                model(x.to(device))
        model.eval()

        # Trainable (unfused) weights — fine-tune from these.
        torch.save({"subnet_config": arch, "f1_score": f1, "latency_ms": lat, "state_dict": model.state_dict()},
                   os.path.join(best_models_dir, f"pareto_model_{idx + 1}_lat{lat:.1f}.pth"))

        # Deploy weights — RepConv folded. Load with: StandaloneSubnet(cfg, arch).fuse().load_state_dict(...)
        model.fuse()
        torch.save({"subnet_config": arch, "state_dict": model.state_dict()},
                   os.path.join(best_models_dir, f"pareto_model_{idx + 1}_lat{lat:.1f}_deploy.pth"))

    with open(os.path.join(config.save_dir, "pareto_front.json"), "w") as f:
        json.dump({"architectures": pareto_architectures, "metrics": pareto_metrics}, f, indent=2)
    print(f"All Pareto-optimal models saved to {best_models_dir}")


if __name__ == "__main__":
    main()