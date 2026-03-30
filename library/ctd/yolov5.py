"""YOLOv5 model architecture vendored from manga-image-translator (GPL-3.0).

Source: github.com/zyddnys/manga-image-translator
Original: github.com/ultralytics/yolov5 (GPL-3.0)
"""

import math
import warnings
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from .utils import (
    check_anchor_order,
    check_version,
    fuse_conv_and_bn,
    initialize_weights,
    make_divisible,
)


# --- common.py modules ---


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        if isinstance(act, bool):
            self.act = (
                nn.SiLU()
                if act is True
                else (act if isinstance(act, nn.Module) else nn.Identity())
            )
        elif isinstance(act, str):
            if act == "leaky":
                self.act = nn.LeakyReLU(0.1, inplace=True)
            elif act == "relu":
                self.act = nn.ReLU(inplace=True)
            else:
                self.act = None

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, act=True):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c_, c2, 3, 1, g=g, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.SiLU()
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n))
        )

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act=True):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c1, c_, 1, 1, act=act)
        self.cv3 = Conv(2 * c_, c2, 1, act=act)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, e=1.0, act=act) for _ in range(n))
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class TransformerLayer(nn.Module):
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)
        self.tr = nn.Sequential(
            *(TransformerLayer(c2, num_heads) for _ in range(num_layers))
        )
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class SPP(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        return self.conv(
            torch.cat(
                [
                    x[..., ::2, ::2],
                    x[..., 1::2, ::2],
                    x[..., ::2, 1::2],
                    x[..., 1::2, 1::2],
                ],
                1,
            )
        )


class GhostConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),
            GhostConv(c_, c2, 1, 1, act=False),
        )
        self.shortcut = (
            nn.Sequential(
                DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)
            )
            if s == 2
            else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        return x.view(b, c * s * s, h // s, w // s)


class Expand(nn.Module):
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()
        s = self.gain
        x = x.view(b, s, s, c // s**2, h, w)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        return x.view(b, c // s**2, h * s, w * s)


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


# --- yolo.py ---


class Detect(nn.Module):
    stride = None
    onnx_dynamic = False

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        self.anchor_grid = [torch.zeros(1)] * self.nl
        self.register_buffer(
            "anchors", torch.tensor(anchors).float().view(self.nl, -1, 2)
        )
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.inplace = inplace

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )
            if not self.training:
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[
                        i
                    ]
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                else:
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, "1.10.0"):
            yv, xv = torch.meshgrid(
                [torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing="ij"
            )
        else:
            yv, xv = torch.meshgrid(
                [torch.arange(ny, device=d), torch.arange(nx, device=d)]
            )
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (
            (self.anchors[i].clone() * self.stride[i])
            .view((1, self.na, 1, 1, 2))
            .expand((1, self.na, ny, nx, 2))
            .float()
        )
        return grid, anchor_grid


class Model(nn.Module):
    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None):
        super().__init__()
        self.out_indices = None
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml

            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)

        ch = self.yaml["ch"] = self.yaml.get("ch", ch)
        if nc and nc != self.yaml["nc"]:
            self.yaml["nc"] = nc
        if anchors:
            self.yaml["anchors"] = round(anchors)
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])
        self.names = [str(i) for i in range(self.yaml["nc"])]
        self.inplace = self.yaml.get("inplace", True)

        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256
            m.inplace = self.inplace
            m.stride = torch.tensor(
                [s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))]
            )
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()
        initialize_weights(self)

    def forward(self, x, augment=False, profile=False, visualize=False, detect=False):
        return self._forward_once(x, profile, visualize, detect=detect)

    def _forward_once(self, x, profile=False, visualize=False, detect=False):
        y, _dt = [], []
        z = []
        for ii, m in enumerate(self.model):
            if m.f != -1:
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )
            x = m(x)
            y.append(x if m.i in self.save else None)
            if self.out_indices is not None:
                if m.i in self.out_indices:
                    z.append(x)
        if self.out_indices is not None:
            if detect:
                return x, z
            else:
                return z
        else:
            return x

    def _initialize_biases(self, cf=None):
        m = self.model[-1]
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += (
                math.log(0.6 / (m.nc - 0.999999))
                if cf is None
                else torch.log(cf / cf.sum())
            )
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def fuse(self):
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, "bn")
                m.forward = m.forward_fuse
        return self

    def _apply(self, fn):
        self = super()._apply(fn)
        m = self.model[-1]
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


def parse_model(d, ch):
    anchors, nc, gd, gw = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
    )
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    no = na * (nc + 5)

    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except NameError:
                pass

        n = max(round(n * gd), 1) if n > 1 else n
        if m in [
            Conv,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            Focus,
            BottleneckCSP,
            C3,
            C3TR,
            C3SPP,
            C3Ghost,
        ]:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        np_ = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type, m_.np = i, f, t, np_
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


@torch.no_grad()
def load_yolov5_ckpt(
    weights, map_location="cpu", fuse=True, inplace=True, out_indices=[1, 3, 5, 7, 9]
):
    if isinstance(weights, str):
        ckpt = torch.load(weights, map_location=map_location)
    else:
        ckpt = weights

    model = Model(ckpt["cfg"])
    model.load_state_dict(ckpt["weights"], strict=True)

    if fuse:
        model = model.float().fuse().eval()
    else:
        model = model.float().eval()

    for m in model.modules():
        if type(m) in [
            nn.Hardswish,
            nn.LeakyReLU,
            nn.ReLU,
            nn.ReLU6,
            nn.SiLU,
            Detect,
            Model,
        ]:
            m.inplace = inplace
            if type(m) is Detect:
                if not isinstance(m.anchor_grid, list):
                    delattr(m, "anchor_grid")
                    setattr(m, "anchor_grid", [torch.zeros(1)] * m.nl)
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()
    model.out_indices = out_indices
    return model
