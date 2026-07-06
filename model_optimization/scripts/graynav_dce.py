from __future__ import annotations

import torch.nn as nn
from ultralytics.nn.modules import Conv


class DCE(nn.Module):
    """Deployment-friendly directional context enhancement for YOLO neck features.

    The block approximates coordinate/directional context with asymmetric
    convolutions only. It intentionally avoids pooling, split, transpose,
    sigmoid, softmax, div and sqrt so the exported graph only adds Conv, BN,
    ReLU and Add style operators.
    """

    def __init__(self, c: int, r: int = 4) -> None:
        super().__init__()
        hidden = max(int(c) // int(r), 8)
        self.reduce = Conv(int(c), hidden, k=1, s=1)
        self.hconv = Conv(hidden, hidden, k=(1, 7), s=1, p=(0, 3))
        self.vconv = Conv(hidden, hidden, k=(7, 1), s=1, p=(3, 0))
        self.act = nn.ReLU(inplace=True)
        self.expand = Conv(hidden, int(c), k=1, s=1, act=False)
        self._zero_init_expand()

    def _zero_init_expand(self) -> None:
        """Initialize the residual branch as exact zero for identity startup."""
        nn.init.zeros_(self.expand.conv.weight)
        if self.expand.conv.bias is not None:
            nn.init.zeros_(self.expand.conv.bias)
        if hasattr(self.expand, "bn"):
            nn.init.ones_(self.expand.bn.weight)
            nn.init.zeros_(self.expand.bn.bias)

    def forward(self, x):
        """Enhance a feature map with horizontal and vertical residual context."""
        z = self.reduce(x)
        u = self.act(self.hconv(z) + self.vconv(z))
        return x + self.expand(u)


def register_ultralytics_dce() -> None:
    """Expose DCE to Ultralytics YAML parsing and checkpoint loading."""
    import ultralytics.nn.tasks as tasks

    setattr(tasks, "DCE", DCE)
