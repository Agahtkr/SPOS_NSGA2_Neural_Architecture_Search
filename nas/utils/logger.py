# utils/logger.py
import os
from torch.utils.tensorboard import SummaryWriter

class ExperimentLogger:
    def __init__(self, config):
        """TensorBoard Writer'ı başlatır ve logları config.save_dir altındaki 'logs' klasörüne yazar."""
        self.log_dir = os.path.join(config.save_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def log_epoch_metrics(self, epoch, avg_total_loss, avg_box_loss, avg_cls_loss):
        """Eğitim kayıplarını (Loss) TensorBoard'a yazar."""
        self.writer.add_scalar("Loss/Total", avg_total_loss, epoch)
        self.writer.add_scalar("Loss/Box", avg_box_loss, epoch)
        self.writer.add_scalar("Loss/Class", avg_cls_loss, epoch)

    def log_validation_f1(self, epoch, f1_score):
        """Validasyon sonucunu TensorBoard'a yazar."""
        self.writer.add_scalar("Metrics/F1_Score", f1_score, epoch)

    def log_learning_rate(self, epoch, lr):
        """O anki öğrenme oranını grafiğe ekler."""
        self.writer.add_scalar("Hyperparameters/Learning_Rate", lr, epoch)

    def close(self):
        """Eğitim bitince dosya yazıcısını güvenle kapatır."""
        self.writer.close()