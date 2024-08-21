IGNORE_LABEL_ID = -100


from enum import Enum, unique
from .loss import bpr_loss, info_nce_loss

@unique
class EnumTokenRole(Enum):
    CTX = "CTX"
    RET = "RET"
    GEN = "GEN"

    def __eq__(self, other):
        if isinstance(other, EnumContrastiveLoss):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return False

@unique
class EnumContrastiveLoss(Enum):
    BPR = "BPR"
    InfoNCE = "InfoNCE"

    def __eq__(self, other):
        if isinstance(other, EnumContrastiveLoss):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return False

    @classmethod
    def is_valid(cls, value):
        return value in cls.__members__

    @classmethod
    def to_list(cls):
        return list(cls.__members__.keys())

    @classmethod
    def get_loss_mapping(cls) -> dict:
        return {
            cls.BPR: bpr_loss,
            cls.BPR.value: bpr_loss,
            cls.InfoNCE: info_nce_loss,
            cls.InfoNCE.value: info_nce_loss
        }

if __name__ == '__main__':
    print(EnumContrastiveLoss.BPR == 'BPR')
    # print(EnumContrastiveLoss.__members__)
    # print(EnumContrastiveLoss.BPR)
    # print(EnumContrastiveLoss.is_valid('BPR'))
    # print(EnumContrastiveLoss.to_list())