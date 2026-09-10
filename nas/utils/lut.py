import time
import torch
from models.operations import OPS
from models.supernet import stage_schedule


class LUTManager:
    """
    Hedef donanım uyumlu LUT (Look-Up Table) Yöneticisi.
    Not: Gecikme ölçümlerini geliştirme yaptığın CPU yerine,
    modeli çalıştıracağın hedef donanımda (target hardware) koşturmalısın.
    """

    def __init__(self, config, device="cpu", warmup=10, iters=100):
        self.config = config
        self.table = {}
        self._profile(device, warmup, iters)

    @torch.no_grad()
    def _profile(self, device, warmup, iters):
        print(f"Profiling hardware latency for LUT on device: {device}...")
        schedule, _ = stage_schedule(self.config)

        # Yeni 4x küçültmeli stem yapısına uyumlu başlangıç çözünürlüğü (160x160)
        h = w = self.config.input_resolution // 4

        for layer_idx, (in_c, out_c, stride) in enumerate(schedule):
            self.table[layer_idx] = {}
            dummy = torch.randn(1, in_c, h, w, device=device)

            for op_idx, builder in enumerate(OPS):
                op = builder(in_c, out_c, stride=stride).to(device)
                if hasattr(op, "fuse_weights"):
                    op.fuse_weights()
                op.eval()

                for _ in range(warmup):
                    op(dummy)
                if device != "cpu":
                    torch.cuda.synchronize()

                start = time.perf_counter()
                for _ in range(iters):
                    op(dummy)
                if device != "cpu":
                    torch.cuda.synchronize()

                self.table[layer_idx][op_idx] = (time.perf_counter() - start) / iters * 1000

            if stride == 2:
                h //= 2
                w //= 2

        print("LUT profiling complete.")

    def get_subnet_latency(self, chromosome):
        return sum(self.table[i][op] for i, op in enumerate(chromosome))