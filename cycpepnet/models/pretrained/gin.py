import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from .base_pretrained_gnn import _GINConvLayer, _PretrainedGNN

# Architecture constants (must match snap-stanford/pretrain-gnns checkpoint layout)
_HIDDEN_DIM         = 300
_NUM_LAYERS         = 5
_NUM_ATOM_TYPES     = 120
_NUM_CHIRALITY_TAGS = 3
_NUM_BOND_TYPES     = 6
_NUM_BOND_DIRS      = 3


# =============================================
#                  ENCODER
# =============================================

# 5-layer GIN encoder matching the snap-stanford/pretrain-gnns state dict
class _GINEncoder(nn.Module):

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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:

        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        for i, (gnn, bn) in enumerate(zip(self.gnns, self.batch_norms)):
            h = gnn(h, edge_index, edge_attr)
            h = bn(h)
            if i < _NUM_LAYERS - 1:
                h = F.relu(h)
            if self.drop_ratio > 0:
                h = F.dropout(h, p=self.drop_ratio, training=self.training)

        return global_mean_pool(h, batch)


# =============================================
#              MODEL
# =============================================

# GIN fine-tuned for molecular property prediction (Hu et al., ICLR 2020)
class PretrainedGINModel(_PretrainedGNN):

    _CHECKPOINT_URL  = "https://raw.githubusercontent.com/snap-stanford/pretrain-gnns/8b20528a83b8869ce16451305b32c827258d19a3/chem/model_gin/supervised_contextpred.pth"
    _CACHE_FILE      = "gin_supervised_contextpred.pth"
    _ARCHITECTURE    = "GIN-5L-300d-JKlast-contextpred"
    _NAME            = "Pretrained-GIN"
    _ATOM_IDX_MAX    = _NUM_ATOM_TYPES - 1
    _ADD_HS          = False
    _ENCODER_CLS     = _GINEncoder
    _ENCODER_OUT_DIM = _HIDDEN_DIM
