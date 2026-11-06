# PeptideCLM-2 (MLM+MTR, small, ~32M parameters): BERT-style encoder pretrained on >100M
# molecules via a dual MLM+MTR objective (Feller et al., 2026, bioRxiv).

import sys
import importlib.util
from types import ModuleType
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from .base_pretrained import _PretrainedModel

# PeptideCLM-2 encoder with a regression/classification head fine-tuned on SMILES
class PeptideCLM2Model(_PretrainedModel):

    _MODEL_ID           = "aaronfeller/PeptideMLM-MTR_sm"
    _MODEL_REVISION     = "154d4c69f03c025a4f70e16bb2a39f3a08700628"
    _TRUST_REMOTE_CODE  = True

    def __init__(self, max_length: int = 768, **kwargs):
        super().__init__(max_length=max_length, **kwargs)

    @property
    def name(self) -> str:
        return "PeptideCLM-2"

    # Load the upstream checkpoint, importing its own config/model modules from the download
    def _load_encoder(self) -> tuple:

        local_path = Path(snapshot_download(self._MODEL_ID, revision=self._MODEL_REVISION))

        _PKG = "_aaronfeller_peptide_clm2"

        if _PKG not in sys.modules:
            pkg = ModuleType(_PKG)
            pkg.__path__    = [str(local_path)]
            pkg.__package__ = _PKG
            sys.modules[_PKG] = pkg

            spec_cfg = importlib.util.spec_from_file_location(f"{_PKG}.config", local_path / "config.py")
            mod_cfg  = importlib.util.module_from_spec(spec_cfg)
            mod_cfg.__package__ = _PKG
            sys.modules[f"{_PKG}.config"] = mod_cfg
            spec_cfg.loader.exec_module(mod_cfg)

            spec_model = importlib.util.spec_from_file_location(f"{_PKG}.ChemPepMTR", local_path / "ChemPepMTR.py")
            mod_model  = importlib.util.module_from_spec(spec_model)
            mod_model.__package__ = _PKG
            sys.modules[f"{_PKG}.ChemPepMTR"] = mod_model
            spec_model.loader.exec_module(mod_model)

        mod_cfg   = sys.modules[f"{_PKG}.config"]
        mod_model = sys.modules[f"{_PKG}.ChemPepMTR"]

        config  = mod_cfg.model_config.from_pretrained(str(local_path))
        encoder = mod_model.MLM_model(config)

        state_dict = load_file(str(local_path / "model.safetensors"))
        encoder.load_state_dict(state_dict)

        return encoder, config.embed_dim

    # PeptideCLM-2 already returns a mean-pooled representation
    def _pool_representation(self, out, attention_mask: torch.Tensor) -> torch.Tensor:
        return out["mean_pool"]
