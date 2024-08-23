from .dataset import AutoDataset
from config import OneGenConfig
from .dataset_utils import padding_item

class BaseDataCollator:
    def __init__(self):
        raise NotImplementedError()

    def __call__(self):
        raise NotImplementedError()

    def random_select(self):
        pass

class AutoDataCollator:
    def __init__(
        self,
        dataset:AutoDataset,
        onegen_config: OneGenConfig
    ):
        raise NotImplementedError()
    
    def __call__(self, instances: List[tuple]):
        """
        for each instance in instances:
            instance: Tuple[int, List[List], List[List]]
            instance = [
                (
                    idx,                                    # index in the AutoDataset
                    positive_id: [[uid, uid], [uid, uid], ...],  # positive index
                    corresponding_neg_qid: [
                        [uid, uid, uid, uid, uid], 
                        [uid, uid, uid, uid, uid],
                        ...
                    ]
                ),
                ...
            ]
        
        Packing Sequence:
            Generation Only (GEN+CTX)
            Hybrid (RET+GEN+CTX)
            Positive (RET+CTX)
            Negative (RET+CTX)
        if there are no positive examples for a case from Hybrid, this case will be converted to Generation-Only

        Our goal is:
            Step 1. Make sure there are no overlap for negative in a batch and there are no overlap between the positive and negative.
            Step 2. Classify the each instance in instances. Generation Only or Hybrid
            Step 3. Get index and flag
            Step 4. padding
            Step 5. Return
        """
    