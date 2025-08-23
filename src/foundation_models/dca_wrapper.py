import torch
import os
import torch.nn as nn
from .lightning_task import LightningTask
from .DCA.dca_seg import dca_base_patch16


class DCASegmentation(LightningTask):
    def __init__(self, args, model_config, data_config) -> None:
        super().__init__(args, model_config, data_config)
        self.model_config = model_config
        self.data_config = data_config

        self.patch_embedding = dca_base_patch16()

        self.criterion = nn.MSELoss()

        self.load_stat_dict()

        if model_config.freeze_teacher:
            self.freeze(self.patch_embedding.teacher)

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
        out_teacher, out_student = outputs[0], outputs[1]
        return self.criterion(out_teacher, out_student)
    
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