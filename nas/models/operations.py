# models/operations.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        # Projection needed whenever channels differ OR spatial resolution
        # changes (stage-transition layers with stride=2).
        self.needs_projection = (in_c != out_c) or (stride != 1)
        if self.needs_projection:
            self.proj = nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        if self.needs_projection:
            return self.proj(x)
        return x


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class MBConv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, expansion_ratio, stride=1, use_se=False):
        super().__init__()
        # Residual only valid when shape is fully preserved (same channels,
        # no spatial downsampling) - a strided/width-changing block cannot
        # be added back to its own input.
        self.use_residual = (in_c == out_c and stride == 1)
        hidden_dim = in_c * expansion_ratio
        padding = kernel_size // 2

        layers = []
        if expansion_ratio != 1:
            layers.extend([
                nn.Conv2d(in_c, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride=stride,
                      padding=padding, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])

        if use_se:
            layers.append(SqueezeExcitation(hidden_dim, max(1, in_c // 4)))

        layers.extend([
            nn.Conv2d(hidden_dim, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        ])

        self.op = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.op(x)
        return self.op(x)


class CSPMBConv(nn.Module):
    """
    CSP-wrapped MBConv. Only half the channels go through the expensive
    MBConv path. When stride == 2, the untouched half must also be spatially
    downsampled (via avg pooling) before concatenation, since the two paths
    have to match spatial size to be concatenated.
    """

    def __init__(self, in_c, out_c, kernel_size, expansion_ratio, stride=1, use_se=False):
        super().__init__()
        self.split_c = in_c // 2
        self.rem_c = in_c - self.split_c

        self.main_block = MBConv(
            in_c=self.split_c, out_c=self.split_c, kernel_size=kernel_size,
            expansion_ratio=expansion_ratio, stride=stride, use_se=use_se
        )

        self.skip_pool = nn.AvgPool2d(kernel_size=stride, stride=stride) if stride != 1 else nn.Identity()

        self.transition = nn.Sequential(
            nn.Conv2d(self.split_c + self.rem_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x):
        x1, x2 = torch.split(x, [self.split_c, self.rem_c], dim=1)
        x1 = self.main_block(x1)
        x2 = self.skip_pool(x2)
        out = torch.cat([x1, x2], dim=1)
        return self.transition(out)




class FusedMBConv(nn.Module):
    """
    EfficientNetV2 tarzı Fused-MBConv.
    1x1 genişletme ve Depthwise (derinlemesine) konvolüsyonu tek bir standart kxk konvolüsyona dönüştürür.
    Bellek darboğazı (memory-bandwidth) yaşayan uç cihazlarda (edge devices) çok daha hızlıdır.
    """

    def __init__(self, in_c, out_c, kernel_size, expansion_ratio, stride=1, use_se=False):
        super().__init__()
        self.use_residual = (in_c == out_c and stride == 1)
        hidden_dim = in_c * expansion_ratio
        padding = kernel_size // 2

        layers = []
        # 1. Expand (Genişlet) ve Feature Extraction (Özellik Çıkar) işlemlerini
        # standart bir Conv2d ile aynı anda yapıyoruz (Depthwise iptal)
        layers.extend([
            nn.Conv2d(in_c, hidden_dim, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])

        # 2. Squeeze-and-Excitation
        if use_se:
            layers.append(SqueezeExcitation(hidden_dim, max(1, in_c // 4)))

        # 3. Pointwise Projection (Daraltma)
        layers.extend([
            nn.Conv2d(hidden_dim, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        ])

        self.op = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.op(x)
        return self.op(x)


class RepConv(nn.Module):
    """
    RepVGG / YOLOv6-style re-parameterizable block.
    Train: 3x3-BN + 1x1-BN (+ BN identity). Deploy: one 3x3 conv with bias.
    fuse_weights() performs the *real* algebraic fold.
    """

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.deploy = False
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride

        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.identity = nn.BatchNorm2d(out_c) if (in_c == out_c and stride == 1) else None

        self.act = nn.ReLU6(inplace=True)
        self.fused_conv = None

    def forward(self, x):
        if self.deploy:
            return self.act(self.fused_conv(x))

        out = self.branch_3x3(x) + self.branch_1x1(x)
        if self.identity is not None:
            out = out + self.identity(x)
        return self.act(out)

    @staticmethod
    def _fuse_conv_bn(kernel, bn):
        """conv(no bias) -> BN  ==  conv(kernel*s, bias=beta - mean*s), s = gamma/sqrt(var+eps)."""
        std = (bn.running_var + bn.eps).sqrt()
        scale = bn.weight / std
        return kernel * scale.reshape(-1, 1, 1, 1), bn.bias - bn.running_mean * scale

    def get_equivalent_kernel_bias(self):
        k3, b3 = self._fuse_conv_bn(self.branch_3x3[0].weight, self.branch_3x3[1])
        k1, b1 = self._fuse_conv_bn(self.branch_1x1[0].weight, self.branch_1x1[1])
        k1 = F.pad(k1, [1, 1, 1, 1])  # 1x1 -> 3x3 (centre tap)

        kernel, bias = k3 + k1, b3 + b1

        if self.identity is not None:
            # Identity == 3x3 conv whose centre tap is the identity matrix.
            id_kernel = torch.zeros((self.out_c, self.in_c, 3, 3), device=k3.device, dtype=k3.dtype)
            for i in range(self.in_c):
                id_kernel[i, i, 1, 1] = 1.0
            kid, bid = self._fuse_conv_bn(id_kernel, self.identity)
            kernel, bias = kernel + kid, bias + bid

        return kernel, bias

    @torch.no_grad()
    def fuse_weights(self):
        """Fold all branches into a single 3x3 conv. Irreversible; drops training branches.
        To load a fused checkpoint: build the model, call fuse, then load_state_dict."""
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.fused_conv = nn.Conv2d(self.in_c, self.out_c, kernel_size=3, stride=self.stride,
                                    padding=1, bias=True).to(kernel.device)
        self.fused_conv.weight.copy_(kernel)
        self.fused_conv.bias.copy_(bias)

        del self.branch_3x3
        del self.branch_1x1
        if self.identity is not None:
            del self.identity
        self.identity = None
        self.deploy = True


def fuse_repconvs(model):
    """Fuse every RepConv in `model` in place (collect first: fusing mutates the module tree)."""
    reps = [m for m in model.modules() if isinstance(m, RepConv)]
    for m in reps:
        m.fuse_weights()
    return model



# --- SADELEŞTİRİLMİŞ VE OPTİMİZE EDİLMİŞ ARAMA UZAYI (14 Ops) ---
# SE ve CSP çıkarıldı; böylece her operasyon çok daha yoğun ve dengeli güncellenecek.
OPS = []
OP_MAPPING = {}

# 0: Identity
OPS.append(lambda in_c, out_c, stride=1: Identity(in_c, out_c, stride=stride))
OP_MAPPING[0] = {'type': 'Identity', 'k': 0, 'e': 0, 'se': 0}

# 1: RepConv
OPS.append(lambda in_c, out_c, stride=1: RepConv(in_c, out_c, stride=stride))
OP_MAPPING[1] = {'type': 'RepConv', 'k': 3, 'e': 1, 'se': 0}

idx = 2
# MBConv ve FusedMBConv üzerinden çekirdek (3, 5) ve expansion (1, 3, 6) kombinasyonları (SE yok)
for block_class, b_name in [(MBConv, 'MBConv'), (FusedMBConv, 'FusedMBConv')]:
    for k in [3, 5]:
        for e in [1, 3, 6]:
            def make_builder(bc, bk, be):
                return lambda in_c, out_c, stride=1: bc(in_c, out_c, kernel_size=bk, expansion_ratio=be,
                                                        stride=stride, use_se=False)

            OPS.append(make_builder(block_class, k, e))
            OP_MAPPING[idx] = {'type': b_name, 'k': k, 'e': e, 'se': 0}
            idx += 1

# Artık len(OPS) tam olarak 14!