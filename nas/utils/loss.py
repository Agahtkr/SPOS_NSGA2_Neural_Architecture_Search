import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss, complete_box_iou_loss


class TargetAwareYoloLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config.num_classes
        self.lambda_box = 7.5
        self.lambda_cls = 0.5

    def cxcywh_to_xyxy(self, boxes):
        x_c, y_c, w, h = boxes.unbind(dim=-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

    def forward(self, predictions, targets):
        targets = targets.float()
        total_loss_box = torch.tensor(0.0, device=targets.device)
        total_loss_cls = torch.tensor(0.0, device=targets.device)

        # predictions: [P3 (80x80), P4 (40x40), P5 (20x20)]
        for scale_idx, pred in enumerate(predictions):
            pred = pred.float()
            B, _, H, W = pred.shape
            pred = pred.permute(0, 2, 3, 1).contiguous()

            pred_boxes_raw = pred[..., :4]
            pred_logits = pred[..., 4:]

            target_cls = torch.zeros_like(pred_logits)
            target_boxes = torch.zeros_like(pred_boxes_raw)
            obj_mask = torch.zeros((B, H, W), dtype=torch.bool, device=pred.device)

            if len(targets) > 0:
                batch_idx = targets[:, 0].long()
                class_id = targets[:, 1].long()
                boxes = targets[:, 2:6]  # [cx, cy, w, h]

                # --- ÖLÇEK BAZLI HEDEF YÖNLENDİRME (SCALE ROUTING) ---
                # Nesnelerin alanına göre hangi FPN katmanına ait olduğunu seçiyoruz
                box_areas = boxes[:, 2] * boxes[:, 3]
                if scale_idx == 0:  # P3 (Küçük nesneler, örn. alan < 0.05)
                    scale_mask = box_areas < 0.05
                elif scale_idx == 1:  # P4 (Orta nesneler)
                    scale_mask = (box_areas >= 0.05) & (box_areas < 0.2)
                else:  # P5 (Büyük nesneler, alan >= 0.2)
                    scale_mask = box_areas >= 0.2

                # Eğer o ölçeğe ait nesne varsa filtrele
                if scale_mask.sum() > 0:
                    b_idx_s = batch_idx[scale_mask]
                    c_id_s = class_id[scale_mask]
                    boxes_s = boxes[scale_mask]

                    grid_x = (boxes_s[:, 0] * W).long().clamp(0, W - 1)
                    grid_y = (boxes_s[:, 1] * H).long().clamp(0, H - 1)

                    obj_mask[b_idx_s, grid_y, grid_x] = True
                    target_boxes[b_idx_s, grid_y, grid_x] = boxes_s
                    target_cls[b_idx_s, grid_y, grid_x, c_id_s] = 1.0

            loss_box = torch.tensor(0.0, device=pred.device)
            if obj_mask.sum() > 0:
                pos_pred_boxes = torch.sigmoid(pred_boxes_raw[obj_mask])
                pred_xyxy = self.cxcywh_to_xyxy(pos_pred_boxes)
                target_xyxy = self.cxcywh_to_xyxy(target_boxes[obj_mask])
                loss_box = complete_box_iou_loss(pred_xyxy, target_xyxy, reduction="mean")

            # Focal loss'un arkaplanda ezilmesini önlemek için sum/reduction ayarı veya normalize edici kullanılabilir
            loss_cls = sigmoid_focal_loss(inputs=pred_logits, targets=target_cls, alpha=0.25, gamma=2.0, reduction="sum")
            # Batch ve kanal başına normalize et
            num_pos = max(1, obj_mask.sum().item())
            loss_cls = loss_cls / num_pos

            total_loss_box += loss_box
            total_loss_cls += loss_cls

        total_loss = (self.lambda_box * total_loss_box) + (self.lambda_cls * total_loss_cls)

        return total_loss, total_loss_box.detach(), total_loss_cls.detach()