# training/engine.py
import os
import random
import torch
from tqdm import tqdm
from utils.metrics import MetricTracker
from utils.loss import TargetAwareYoloLoss
from utils.logger import ExperimentLogger
from models.operations import OPS


class NASEngine:
    def __init__(self, model, train_loader, val_loader, config, device="cuda", enable_logging=True):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.num_ops = len(OPS)
        self.paths_per_batch = max(1, getattr(config, "paths_per_batch", 1))

        self.use_amp = str(device).startswith("cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

        self.criterion = TargetAwareYoloLoss(config).to(device)
        self.metric_tracker = MetricTracker()
        # Search-time evaluation (run_search.py) does not need a TensorBoard writer.
        self.logger = ExperimentLogger(config) if enable_logging else None

    def close(self):
        if self.logger is not None:
            self.logger.close()
            self.logger = None

    def sample_path(self):
        return [random.randint(0, self.num_ops - 1) for _ in range(self.config.num_layers)]

    def train_supernet_epoch(self, epoch):
        self.model.train()
        epoch_total_loss = epoch_box_loss = epoch_cls_loss = 0.0
        batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.supernet_epochs}")

        for x, targets in pbar:
            x = x.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            batch_total = batch_box = batch_cls = 0.0

            # Accumulate gradients over several random paths so more (layer, op)
            # pairs receive updates per optimizer step. Loss is averaged so the
            # effective LR does not scale with paths_per_batch.
            for _ in range(self.paths_per_batch):
                path = self.sample_path()
                with torch.autocast(device_type="cuda" if self.use_amp else "cpu", enabled=self.use_amp):
                    logits = self.model(x, subnet_config=path)
                    total_loss, loss_box, loss_cls = self.criterion(logits, targets)

                self.scaler.scale(total_loss / self.paths_per_batch).backward()

                batch_total += total_loss.item() / self.paths_per_batch
                batch_box += loss_box.item() / self.paths_per_batch
                batch_cls += loss_cls.item() / self.paths_per_batch

            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_total_loss += batch_total
            epoch_box_loss += batch_box
            epoch_cls_loss += batch_cls
            batches += 1

            pbar.set_postfix({"Total": f"{batch_total:.3f}", "Box": f"{batch_box:.3f}", "Cls": f"{batch_cls:.3f}"})

        if batches > 0 and self.logger is not None:
            self.logger.log_epoch_metrics(
                epoch=epoch,
                avg_total_loss=epoch_total_loss / batches,
                avg_box_loss=epoch_box_loss / batches,
                avg_cls_loss=epoch_cls_loss / batches,
            )
            self.logger.log_learning_rate(epoch, self.optimizer.param_groups[0]["lr"])

    def evaluate_subnet(self, subnet_config):
        """BN recalibration + fast validation pass; BN buffers are snapshotted and restored."""
        subnet_config = list(subnet_config)

        bn_backup = {}
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                bn_backup[name] = {
                    "running_mean": module.running_mean.clone(),
                    "running_var": module.running_var.clone(),
                    "num_batches_tracked": module.num_batches_tracked.clone(),
                    "momentum": module.momentum,
                }
                module.reset_running_stats()
                module.momentum = None  # cumulative average during calibration

        self.model.train()
        with torch.no_grad():
            for i, (x, _) in enumerate(self.train_loader):
                if i >= self.config.bn_calib_batches:
                    break
                _ = self.model(x.to(self.device), subnet_config=subnet_config)

        self.model.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for x, targets in self.val_loader:
                x, targets = x.to(self.device), targets.to(self.device)

                # logits artık [P3, P4, P5] olmak üzere 3 tensör içeren bir liste
                logits = self.model(x, subnet_config=subnet_config)

                for scale_idx, pred_tensor in enumerate(logits):
                    _, _, H, W = pred_tensor.shape
                    preds = torch.argmax(pred_tensor[:, 4:, :, :], dim=1)

                    if len(targets) > 0:
                        batch_idx = targets[:, 0].long()
                        class_id = targets[:, 1].long()
                        boxes = targets[:, 2:6]
                        box_areas = boxes[:, 2] * boxes[:, 3]

                        # Nesneleri alanına göre değerlendirilecekleri katmana yönlendir (loss.py ile aynı mantık)
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

                            all_preds.extend(preds[b_idx_s, grid_y, grid_x].cpu().numpy())
                            all_targets.extend(c_id_s.cpu().numpy())

        # Eğer batch boş gelirse ve metric hesaplanamazsa patlamaması için ufak bir güvenlik
        if len(all_targets) == 0:
            fitness_score = 0.0
        else:
            fitness_score = self.metric_tracker.calculate_f1(all_targets, all_preds)

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                b = bn_backup[name]
                module.running_mean = b["running_mean"]
                module.running_var = b["running_var"]
                module.num_batches_tracked = b["num_batches_tracked"]
                module.momentum = b["momentum"]

        return fitness_score

    def _unwrapped(self):
        return self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

    def save_checkpoint(self, epoch, is_best=False):
        os.makedirs(self.config.save_dir, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self._unwrapped().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(self.config.save_dir, "supernet_last.pth"))
        if is_best:
            torch.save(checkpoint, os.path.join(self.config.save_dir, "supernet_best.pth"))

    def load_checkpoint(self, checkpoint_path):
        if not os.path.isfile(checkpoint_path):
            print("=> No checkpoint found. Starting from scratch...")
            return 0
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self._unwrapped().load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"=> Checkpoint loaded! Resuming training from epoch {start_epoch}...")
        return start_epoch