import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from .base_pretrained_gnn import _GINConvLayer, _PretrainedGNN

# Architecture constants
_HIDDEN_DIM         = 300
_FEAT_DIM           = 512
_NUM_LAYERS         = 5
_NUM_ATOM_TYPES     = 119
_NUM_CHIRALITY_TAGS = 3
_NUM_BOND_TYPES     = 5
_NUM_BOND_DIRS      = 3


# =============================================
#              ENCODER
# =============================================

# 5-layer GIN encoder matching the MolCLR state dict (Wang et al., Nat Mach Intell 2022)
class _MolCLREncoder(nn.Module):

    def __init__(self, drop_ratio: float = 0.0):

        super().__init__()

        self.drop_ratio = drop_ratio

        self.x_embedding1 = nn.Embedding(_NUM_ATOM_TYPES,     _HIDDEN_DIM)
        self.x_embedding2 = nn.Embedding(_NUM_CHIRALITY_TAGS, _HIDDEN_DIM)

        nn.init.xavier_uniform_(self.x_embedding1.weight)
        nn.init.xavier_uniform_(self.x_embedding2.weight)

        self.gnns        = nn.ModuleList([
            _GINConvLayer(_HIDDEN_DIM, _NUM_BOND_TYPES, _NUM_BOND_DIRS) for _ in range(_NUM_LAYERS)
        ])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(_HIDDEN_DIM) for _ in range(_NUM_LAYERS)])

        self.feat_lin    = nn.Linear(_HIDDEN_DIM, _FEAT_DIM)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:

        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        for i, (gnn, bn) in enumerate(zip(self.gnns, self.batch_norms)):
            h = gnn(h, edge_index, edge_attr)
            h = bn(h)
            if i < _NUM_LAYERS - 1:
                h = F.relu(h)
            if self.drop_ratio > 0:
                h = F.dropout(h, p=self.drop_ratio, training=self.training)

        g = global_mean_pool(h, batch)

        return self.feat_lin(g)


# =============================================
#              MODEL
# =============================================

# Pretrained MolCLR fine-tuned for molecular property prediction
class MolCLRModel(_PretrainedGNN):

    _CHECKPOINT_URL     = "https://raw.githubusercontent.com/yuyangw/MolCLR/3d3bc1912be27b0c97435fd9134f4d4c73d4c5ab/ckpt/pretrained_gin/checkpoints/model.pth"
    _CACHE_FILE         = "molclr_pretrained_gin.pth"
    _ARCHITECTURE       = "GIN-5L-300d-feat512-simclr"
    _NAME               = "MolCLR"
    _ATOM_IDX_MAX       = _NUM_ATOM_TYPES - 2
    _ADD_HS             = True
    _ENCODER_CLS        = _MolCLREncoder
    _ENCODER_OUT_DIM    = _FEAT_DIM
    _STRIP_KEY_PREFIXES = ("out_lin.",)
