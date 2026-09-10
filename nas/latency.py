import torch
import time
import os
from ultralytics import YOLO

# CPU çekirdeklerini sınırlamak, işletim sisteminin arka plan
# işlemlerinden kaynaklı dalgalanmaları (varyans) azaltır.
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
torch.set_num_threads(4)

# 1. Ultralytics modeli ve ağırlıkları otomatik olarak yükler
model_path = "/home/aisoft/Aisoft/Gumus-su/0.5L-area/model/paket_13_08_v2_cls_kapak_odakli/yolo26n_baseline_best.pt"
yolo_wrapper = YOLO(model_path)

# 2. NAS algoritmasıyla tamamen adil kıyaslama yapabilmek için
# ekstra işlemleri atlayıp saf PyTorch ağını çekiyoruz.
model = yolo_wrapper.model.to("cpu")
model.eval()

# 3. Giriş tensörü (Siyah beyaz - in_channels=1)
dummy_input = torch.randn(1, 3, 640, 640).to("cpu")

print("İşlemci (CPU) ısınma turları atılıyor...")
with torch.no_grad():
    for _ in range(50):
        _ = model(dummy_input)

print("Hız testi başlıyor...")
iterations = 500
total_time = 0.0

with torch.no_grad():
    for _ in range(iterations):
        start_time = time.perf_counter()
        _ = model(dummy_input)
        total_time += (time.perf_counter() - start_time)

avg_time_ms = (total_time / iterations) * 1000
fps = 1000 / avg_time_ms if avg_time_ms > 0 else 0

print("="*40)
print(f"Toplam Test Süresi: {total_time:.2f} saniye")
print(f"Ortalama Çıkarım Süresi (Latency): {avg_time_ms:.2f} ms")
print(f"Saniyedeki Kare Hızı (FPS): {fps:.1f}")
print("="*40)