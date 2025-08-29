import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from .lightning_task import LightningTask
from .DCA.dca_seg import dca_base_patch16


class DCASegmentation(LightningTask):
    def __init__(self, args, model_config, data_config) -> None:
        super().__init__(args, model_config, data_config)
        self.model_config = model_config
        self.data_config = data_config

        self.patch_embedding = dca_base_patch16()

        # self.criterion = nn.MSELoss()

        self.load_stat_dict()

        if model_config.freeze_teacher:
            self.freeze(self.patch_embedding.teacher)

    def compute_affinity_matrix(self, x):
        """
        Compute affinity matrix of feature `x`

        Args: 
            x: tensor of shape (B,N,D), where `N` is patch numbers, `D` is featue dimension.
        """
        x_norm = F.normalize(x, p=2, dim=2, eps=1e-6) # l2-normalized features of `x` 
        affinity_matrix = torch.einsum('bnd,bmd->bnm', x_norm, x_norm)
        return affinity_matrix

    def load_stat_dict(self):
        # load weights from merged checkpoint
        pretrained_path = self.model_config.get("pretrained_path", None)
        if not pretrained_path:
            raise KeyError(f"'pretrained_path' must be provided in 'src/configs/model/dca_*.yaml'")
        
        ckpt = torch.load(pretrained_path)
        missing_keys, unexpected_keys = self.patch_embedding.load_state_dict(ckpt, strict=False)

        if missing_keys or unexpected_keys:
            print(f"Model missing keys: {missing_keys}, unexpected keys: {unexpected_keys}")

    def forward(self, batch):
        out = self.patch_embedding(batch, self.data_config.band_wavelengths)
        return out

    def loss(self, outputs, labels):
        loss_type = self.model_config.get("loss_type", "affinity_l2")
        out_teacher, out_student = outputs[0], outputs[1]
        if loss_type is "direct_l2":
            return F.mse_loss(out_teacher, out_student)
        elif loss_type is "affinity_l2":
            affinity_teacher = self.compute_affinity_matrix(out_teacher)
            affinity_student = self.compute_affinity_matrix(out_student)
            return F.mse_loss(affinity_teacher, affinity_student)
        elif loss_type is "random_proj_l2":
            b, n, d = out_teacher.shape
            random_proj = torch.randn(d, d, device=out_teacher.device)
            proj_teacher = torch.einsum('bnd,df->bnf', out_teacher, random_proj)
            proj_student = torch.einsum('bnd,df->bnf', out_student, random_proj)
            return F.mse_loss(proj_teacher, proj_student)
        elif loss_type is "mixed":
            pass
        else:
            raise ValueError(f"`loss_type` {loss_type} not defined")
        
    def log_metrics(self, outputs, targets, prefix="train"):
        loss = self.loss(outputs, targets)
        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    def params_to_optimize(self):
        return (
            list(self.patch_embedding.student.parameters())
        )


def DCAModel(args, model_config, data_config):
    if args.task == "segmentation":
        return DCASegmentation(args, model_config, data_config)
    else:
        raise NotImplementedError(f"task {args.tasks} not implemented")