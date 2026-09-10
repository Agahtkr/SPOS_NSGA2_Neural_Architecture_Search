# models/supernet.py
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.operations import OPS, fuse_repconvs

WIDTH_MULTS = [0.75, 1.0, 1.25]


def stage_schedule(config, width_mult=1.0):
    total_layers = config.num_layers
    transitions = {total_layers // 4, (total_layers // 4) * 2, (total_layers // 4) * 3}

    schedule, in_c = [], int(config.base_channels * width_mult)
    for layer_idx in range(total_layers):
        stride = 2 if layer_idx in transitions else 1
        out_c = int((in_c * 2 if stride == 2 else in_c) * width_mult) if layer_idx > 0 else int(
            config.base_channels * width_mult)
        # Kanal sayılarını 8'in katı yapmak donanım hızlandırması için önemlidir
        in_c = make_divisible(in_c, 8)
        out_c = make_divisible(out_c, 8)

        schedule.append((in_c, out_c, stride))
        in_c = out_c

    sorted_transitions = sorted(list(transitions))
    feature_layers = [sorted_transitions[0], sorted_transitions[1], total_layers - 1]
    return schedule, feature_layers


def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _conv_bn(in_c, out_c, k, act=False):
    layers = [nn.Conv2d(in_c, out_c, k, padding=k // 2, bias=False), nn.BatchNorm2d(out_c)]
    if act:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)


class LightFPN(nn.Module):
    def __init__(self, c3, c4, c5):
        super().__init__()
        self.out_c = c3
        self.lat_c5 = _conv_bn(c5, c3, 1)
        self.lat_c4 = _conv_bn(c4, c3, 1)
        self.smooth_p4 = _conv_bn(c3, c3, 3, act=True)
        self.smooth_p3 = _conv_bn(c3, c3, 3, act=True)

    def forward(self, c3, c4, c5):
        p5 = self.lat_c5(c5)
        p4 = self.smooth_p4(self.lat_c4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest"))
        p3 = self.smooth_p3(c3 + F.interpolate(p4, size=c3.shape[-2:], mode="nearest"))
        # Sadece P3 yerine 3 ölçeği de döndür (8x, 16x, 32x)
        return [p3, p4, p5]


class MultiScaleHead(nn.Module):
    def __init__(self, in_c, num_classes, prior_prob=0.01):
        super().__init__()
        # Farklı ölçekler için paylaşımlı (shared) başlık ağırlıkları
        self.head = nn.Conv2d(in_c, 4 + num_classes, kernel_size=1)
        nn.init.constant_(self.head.bias[4:], -math.log((1 - prior_prob) / prior_prob))

    def forward(self, features):
        # 3 farklı FPN çıkışını ayrı ayrı başlık üzerinden geçirip listele
        outputs = []
        for feat in features:
            outputs.append(self.head(feat))
        return outputs

def _make_stem(config):
    """
    Standart YOLO tarzı kök (Stem):
    Girdiyi ilk başta 4x küçülterek (iki adet stride=2 konvolüsyon ile)
    640x640 boyutunu doğrudan 160x160'a indirir ve modeli rahatlatır.
    """
    return nn.Sequential(
        nn.Conv2d(config.in_channels, config.base_channels, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(config.base_channels),
        nn.ReLU6(inplace=True),
        nn.Conv2d(config.base_channels, config.base_channels, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(config.base_channels),
        nn.ReLU6(inplace=True)
    )

def _make_head(in_c, num_classes, prior_prob=0.01):
    head = nn.Conv2d(in_c, 4 + num_classes, kernel_size=1)
    # Focal-loss prior: start class logits at p=0.01 so the (overwhelmingly
    # negative) grid does not dominate the first epochs.
    nn.init.constant_(head.bias[4:], -math.log((1 - prior_prob) / prior_prob))
    return head


class MixedOp(nn.Module):
    """
    Her katman için hem farklı operasyonları hem de
    farklı kanal genişliği çarpanlarını barındıran esnek yapı.
    """

    def __init__(self, in_channels_base, out_channels_base, stride=1):
        super().__init__()
        self.ops = nn.ModuleList()
        self.in_bases = []
        self.out_bases = []

        for mult in WIDTH_MULTS:
            in_c = make_divisible(int(in_channels_base * mult), 8)
            out_c = make_divisible(int(out_channels_base * mult), 8)
            self.in_bases.append(in_c)
            self.out_bases.append(out_c)

            # Her çarpan için operasyonları inşa et
            op_group = nn.ModuleList([op(in_c, out_c, stride=stride) for op in OPS])
            self.ops.append(op_group)

    def forward(self, x, active_op_idx, active_mult_idx):
        # Seçilen kanal çarpanına ve operasyona göre yönlendir
        return self.ops[active_mult_idx][active_op_idx](x)


class SPOSSupernet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_layers = config.num_layers
        self.num_ops = len(OPS)

        schedule, self.feature_layers = stage_schedule(config)
        self.feature_layer_set = set(self.feature_layers)

        self.stem = _make_stem(config)
        self.cells = nn.ModuleList(MixedOp(i, o, stride=s) for i, o, s in schedule)

        c3, c4, c5 = [schedule[i][1] for i in self.feature_layers]
        self.neck = LightFPN(c3, c4, c5)
        self.det_head = MultiScaleHead(self.neck.out_c, config.num_classes)

    def sample_path(self):
        # Her katman için hem operasyon indeksi hem de kanal çarpanı (WIDTH_MULTS) seçilir
        path = []
        for _ in range(self.num_layers):
            op_idx = random.randint(0, self.num_ops - 1)
            mult_idx = random.randint(0, len(WIDTH_MULTS) - 1)
            path.append((op_idx, mult_idx))
        return path

    def forward(self, x, subnet_config=None):
        if subnet_config is None:
            subnet_config = self.sample_path()

        features = []
        x = self.stem(x)
        for layer_idx, cell in enumerate(self.cells):
            cfg = subnet_config[layer_idx]

            # Gelen konfigürasyon ister düz int ister tuple olsun, güvenle ayrıştırıyoruz
            if isinstance(cfg, (tuple, list)):
                op_idx, mult_idx = cfg
            else:
                op_idx = cfg
                mult_idx = 1  # Varsayılan orta kanal çarpanı indeksi (ör. 1.0)

            x = cell(x, op_idx, mult_idx)
            if layer_idx in self.feature_layer_set:
                features.append(x)

        return self.det_head(self.neck(*features))


class StandaloneSubnet(nn.Module):
    """One fixed architecture, weights inherited from the supernet."""

    def __init__(self, config, subnet_config, supernet=None):
        super().__init__()
        schedule, self.feature_layers = stage_schedule(config)
        self.feature_layer_set = set(self.feature_layers)

        self.stem = _make_stem(config)
        self.cells = nn.ModuleList()

        for layer_idx, (cfg, (base_in_c, base_out_c, stride)) in enumerate(zip(subnet_config, schedule)):
            # Genetik algoritmadan gelen tuple'ı (op_idx, mult_idx) ayrıştır
            if isinstance(cfg, (tuple, list)):
                op_idx, mult_idx = cfg
            else:
                op_idx = cfg
                mult_idx = 1  # Varsayılan çarpan indeksi

            mult = WIDTH_MULTS[mult_idx]
            in_c = make_divisible(int(base_in_c * mult), 8)
            out_c = make_divisible(int(base_out_c * mult), 8)

            block = OPS[op_idx](in_c, out_c, stride=stride)
            if supernet is not None:
                # Doğru kanal çarpanı havuzundan ağırlıkları kopyala
                block.load_state_dict(supernet.cells[layer_idx].ops[mult_idx][op_idx].state_dict())
            self.cells.append(block)

        c3, c4, c5 = [schedule[i][1] for i in self.feature_layers]
        self.neck = LightFPN(c3, c4, c5)
        # Eski _make_head yerine yeni çoklu ölçekli başlığı (MultiScaleHead) kullan
        self.det_head = MultiScaleHead(self.neck.out_c, config.num_classes)

        if supernet is not None:
            self.stem.load_state_dict(supernet.stem.state_dict())
            self.neck.load_state_dict(supernet.neck.state_dict())
            self.det_head.load_state_dict(supernet.det_head.state_dict())

    def fuse(self):
        """Deploy-time re-parameterization of every RepConv (irreversible)."""
        fuse_repconvs(self)
        return self

    def forward(self, x):
        features = []
        x = self.stem(x)
        for layer_idx, cell in enumerate(self.cells):
            x = cell(x)
            if layer_idx in self.feature_layer_set:
                features.append(x)
        return self.det_head(self.neck(*features))