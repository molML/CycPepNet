# ChemBERTa-2 MTR-77M: RoBERTa-based encoder pretrained on 77M SMILES via MTR over
# 200 RDKit physicochemical descriptors (Ahmad et al., 2022).

from .base_pretrained import _PretrainedModel

# ChemBERTa-2 encoder with a regression/classification head fine-tuned on SMILES
class ChemBERTa2Model(_PretrainedModel):

    _MODEL_ID       = "DeepChem/ChemBERTa-77M-MTR"
    _MODEL_REVISION = "66b895cab8adebea0cb59a8effa66b2020f204ca"

    @property
    def name(self) -> str:
        return "ChemBERTa-2"
