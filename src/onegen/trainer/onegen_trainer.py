# Part of the code was referenced from https://github.com/ContextualAI/gritlm/

from transformers.trainer_pt_utils import distributed_concat
from transformers.integrations import TensorBoardCallback, rewrite_logs
from transformers import Trainer
import torch.nn as nn

class OneGenTensorBoardCallback(TensorBoardCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):

        if self.tb_writer is None:
            self._init_summary_writer(args)
        
        if hasattr(state, "loss_emb") and hasattr(state, "loss_gen"):
            # Gather & avg across gpus like for actual loss
            # https://github.com/huggingface/transformers/blob/bc72b4e2cdcbc80d5f56731f35dbc9c18b4c8de6/src/transformers/trainer.py#L2257
            if (args.distributed_state is not None and args.distributed_state.distributed_type != "NO") or (
                args.distributed_state is None and args.local_rank != -1):
                state.loss_emb = distributed_concat(state.loss_emb).mean().item()
                state.loss_gen = distributed_concat(state.loss_gen).mean().item()
            else:
                state.loss_emb = state.loss_emb.mean().item()
                state.loss_gen = state.loss_gen.mean().item()
            if state.is_world_process_zero:
                # self._wandb.log({
                #     **rewrite_logs(logs),
                #     "train/global_step": state.global_step,
                #     "train/loss_emb": state.loss_emb,
                #     "train/loss_gen": state.loss_gen,
                # })
                self.tb_writer.add_scalar("contrastive loss", state.loss_emb, state.global_step)
                self.tb_writer.add_scalar("generative loss", state.loss_gen, state.global_step)
            del state.loss_emb
            del state.loss_gen
            self.tb_writer.flush()
        else:
            if not state.is_world_process_zero:
                return

            if self.tb_writer is not None:
                # print("1:", logs)
                logs = rewrite_logs(logs)
                # print("2:", logs)
                for k, v in logs.items():
                    if isinstance(v, (int, float)):
                        self.tb_writer.add_scalar(k, v, state.global_step)
                    else:
                        logger.warning(
                            "Trainer is attempting to log a value of "
                            f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                            "This invocation of Tensorboard's writer.add_scalar() "
                            "is incorrect so we dropped this attribute."
                        )
                self.tb_writer.flush()

class OneGenTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # if is_sagemaker_mp_enabled():
        #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
        #     return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            output = self.compute_loss(model, inputs, True)
            # print(output)
            # assert False
            # loss_emb = loss.
            loss = output[0]
            loss_emb:float = output[1]['cl_loss']
            loss_gen:float = output[1]['npt_loss']
            self.state.loss_emb = getattr(self.state, "loss_emb", torch.tensor(0.0).to(loss.device))
            self.state.loss_gen = getattr(self.state, "loss_gen", torch.tensor(0.0).to(loss.device))
            self.state.loss_emb += loss_emb / self.args.gradient_accumulation_steps
            self.state.loss_gen += loss_gen / self.args.gradient_accumulation_steps

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps
