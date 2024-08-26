
from transformers.trainer_pt_utils import distributed_concat
from transformers.integrations import TensorBoardCallback, rewrite_logs
from transformers import Trainer
import torch.nn as nn


class OneGenTensorBoardCallback(TensorBoardCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):

        if self.tb_writer is None:
            self._init_summary_writer(args)
        
        if hasattr(state, "loss_retrieval") and hasattr(state, "loss_generation"):
            # Gather & avg across gpus like for actual loss
            # https://github.com/huggingface/transformers/blob/bc72b4e2cdcbc80d5f56731f35dbc9c18b4c8de6/src/transformers/trainer.py#L2257
            pass

class OneGenTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        pass