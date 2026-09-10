import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score

from data.dataloader import DataManager
from utils.loss import TargetAwareYoloLoss
from models.supernet import SPOSSupernet, StandaloneSubnet
from configs.search_config import load_config

# BN's default momentum (PyTorch's own default for nn.BatchNorm2d).
# Used to restore normal exponential-moving-average behavior after the
# one-off calibration pass, which deliberately uses momentum=None instead.
BN_DEFAULT_MOMENTUM = 0.1


@torch.no_grad()
def evaluate_f1(model, val_loader, device):
    """Shared eval logic so the baseline check and the per-epoch check
    use IDENTICAL computation."""
    model.eval()
    all_true, all_pred = [], []

    for images, targets in val_loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        if len(targets) == 0:
            continue

        batch_idx = targets[:, 0].long()
        class_id = targets[:, 1].long()
        boxes = targets[:, 2:6]
        box_areas = boxes[:, 2] * boxes[:, 3]

        for scale_idx, pred in enumerate(logits):
            _, _, H, W = pred.shape
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

    model.train()
    if len(all_true) == 0:
        return 0.0
    return f1_score(all_true, all_pred, average='macro', zero_division=0)


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_chromosome = [1, 1, 0, 2, 8, 2, 10, 6, 5, 7, 8, 7]
    print(f"Kromozom {best_chromosome} ile model oluşturuluyor...")

    supernet_path = "/home/aisoft/Aisoft/Gumus-su/0.5L-area/script/nas/experiments/supernet_last.pth"
    if not os.path.exists(supernet_path):
        raise FileNotFoundError(
            f"No trained supernet checkpoint found at {supernet_path}. "
            "This script fine-tunes a searched architecture starting from "
            "its supernet-inherited weights - run train_supernet.py first."
        )

    print(f"Loading supernet weights from {supernet_path}...")
    supernet = SPOSSupernet(config)
    checkpoint = torch.load(supernet_path, map_location="cpu")
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    supernet.load_state_dict(state)

    model = StandaloneSubnet(config, best_chromosome, supernet=supernet).to(device)
    print("=> Backbone cells, stem, neck, and head all inherited from the trained supernet.")

    config.train_img_dir = "/home/aisoft/Aisoft/Gumus-su/0.5L-area/data/dataset_13_08_v2/images/train"
    config.train_lbl_dir = "/home/aisoft/Aisoft/Gumus-su/0.5L-area/data/dataset_13_08_v2/labels/train"
    config.val_image_dir = "/home/aisoft/Aisoft/Gumus-su/0.5L-area/data/dataset_13_08_v2/images/val"
    config.val_lbl_dir = "/home/aisoft/Aisoft/Gumus-su/0.5L-area/data/dataset_13_08_v2/labels/val"

    data_manager = DataManager(config)
    train_loader, val_loader = data_manager.get_dataloaders(
        config.train_img_dir, config.train_lbl_dir,
        config.val_image_dir, config.val_lbl_dir
    )

    print("Recalibrating BatchNorm statistics for this specific architecture...")
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # cumulative average - correct ONLY for this one-off pass
    bn_calib_batches = getattr(config, "bn_calib_batches", 10)
    with torch.no_grad():
        for i, (x, _) in enumerate(train_loader):
            if i >= bn_calib_batches:
                break
            model(x.to(device))

    # FIX (critical - this is the actual cause of the monotonic F1 decline):
    # momentum=None was never switched back to a normal value before the
    # fine-tuning loop below. With momentum=None, BatchNorm keeps using
    # CUMULATIVE averaging (each new batch weighted 1/(n+1), n = batches
    # seen since the last reset) rather than a fixed exponential moving
    # average - so by a few dozen batches into fine-tuning, a new batch's
    # influence on the running stats has decayed to a fraction of a
    # percent and the running stats are effectively FROZEN. Meanwhile the
    # conv weights keep changing (hence train loss still looking fine),
    # so the frozen BN stats drift further from the model's actual,
    # evolving activation distribution every epoch - and since .eval()
    # mode (used for every validation pass) normalizes using those stale
    # running stats, validation F1 degrades a little more each epoch even
    # though nothing about the underlying learning is actually going
    # wrong. That's exactly the smooth, monotonic decline you saw.
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.momentum = BN_DEFAULT_MOMENTUM
    model.eval()

    baseline_f1 = evaluate_f1(model, val_loader, device)
    print(f"=> Baseline F1 immediately after calibration (before fine-tuning): {baseline_f1:.4f}")

    # FIX: this baseline was never actually saved anywhere - if fine-tuning
    # fails to beat it (as happened here), it would otherwise be silently
    # lost even though it's the best result you have. Save it as a
    # checkpoint immediately so it's never at risk.
    os.makedirs(os.path.join(config.save_dir, "subnet_weights"), exist_ok=True)
    baseline_path = os.path.join(config.save_dir, "subnet_weights", "subnet_baseline_calibrated.pth")
    torch.save({
        'epoch': 0,
        'val_f1': baseline_f1,
        'model_state_dict': model.state_dict(),
        'chromosome': best_chromosome
    }, baseline_path)
    print(f"=> Calibrated baseline saved to {baseline_path} (in case fine-tuning doesn't beat it).")

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    criterion = TargetAwareYoloLoss(config).to(device)

    fine_tune_lr = getattr(config, "subnet_finetune_lr", config.learning_rate * 0.01)
    # NOTE: since the model already starts very close to its calibrated
    # optimum (baseline_f1 above), running the FULL supernet_epochs (200)
    # schedule for "fine-tuning" gives a long time for even small residual
    # issues (or ordinary overfitting on a ~30-batch/epoch dataset) to
    # erode a result that was already good. Consider a much shorter
    # dedicated schedule here (e.g. config.subnet_finetune_epochs = 20-30)
    # rather than reusing supernet_epochs, now that the BN bug is fixed.
    finetune_epochs = getattr(config, "subnet_finetune_epochs", config.supernet_epochs)
    optimizer = optim.AdamW(model.parameters(), lr=fine_tune_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    writer = SummaryWriter(log_dir=os.path.join(config.save_dir, "subnet_logs"))

    print(f"\nFine-tuning at LR={fine_tune_lr} for {finetune_epochs} epochs with validation tracking...")

    best_val_f1 = baseline_f1  # FIX: previously started at 0.0, so even a
    # fine-tuned checkpoint WORSE than the calibrated baseline would still
    # get saved as "best" the first epoch it ran. Now fine-tuning has to
    # actually beat the baseline to overwrite anything.

    for epoch in range(1, finetune_epochs + 1):
        model.train()
        total_loss_epoch = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{finetune_epochs}")

        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                predictions = model(images)
                loss, loss_box, loss_cls = criterion(predictions, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss_epoch += loss.item()
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Box": f"{loss_box.item():.4f}",
                "Cls": f"{loss_cls.item():.4f}"
            })

        scheduler.step()
        avg_loss = total_loss_epoch / len(train_loader)
        writer.add_scalar("Train/Loss", avg_loss, epoch)

        current_val_f1 = evaluate_f1(model, val_loader, device)
        writer.add_scalar("Val/Macro_F1", current_val_f1, epoch)
        print(f"Epoch {epoch} - Validation Macro F1: {current_val_f1:.4f} (baseline: {baseline_f1:.4f})")

        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            best_save_path = os.path.join(config.save_dir, "subnet_weights", "subnet_best.pth")

            model_state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()

            torch.save({
                'epoch': epoch,
                'val_f1': best_val_f1,
                'model_state_dict': model_state_dict,
                'chromosome': best_chromosome
            }, best_save_path)
            print(f"--> Yeni en iyi model kaydedildi! F1: {best_val_f1:.4f}")

    if best_val_f1 <= baseline_f1:
        print(f"\nFine-tuning never beat the calibrated baseline ({baseline_f1:.4f}). "
              f"Use subnet_baseline_calibrated.pth - it's your best model.")
    print(f"Eğitim tamamlandı! En iyi doğrulama F1 skoru: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()