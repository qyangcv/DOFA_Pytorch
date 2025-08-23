import os
import argparse
from pathlib import Path
import torch
import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf
from lightning import Trainer

from src.factory import model_registry
from src.datasets.data_module import BenchmarkDataModule


@pytest.fixture()
def dca_config():
    os.environ["PROJECT_DIR"] = os.getenv("PROJECT_DIR", str(Path.cwd().parent))
    config_path = os.path.join("../src/configs/model/dca_seg.yaml")
    model_conf = OmegaConf.load(config_path)
    return model_conf


@pytest.fixture()
def other_args():
    args = argparse.Namespace()
    args.task = "segmentation"
    args.lr = 0.001
    args.weight_decay = 0.0
    args.warmup_epochs = 0
    args.num_gpus = 0
    args.epochs = 1
    args.output_dir = "${oc.env:PROJECT_DIR}/outputs"
    return args


@pytest.fixture()
def data_config():
    data_path = os.path.join("../src/configs/dataset/geobench_cashew.yaml")
    data_conf = OmegaConf.load(data_path)
    return data_conf


@pytest.fixture()
def model(dca_config, other_args, data_config, tmp_path: Path):
    model_name = dca_config.model_type
    model_class = model_registry.get(model_name)
    if model_class is None:
        raise ValueError(f"Model type '{model_name}' not found.")
    return model_class(other_args, dca_config, data_config)


@pytest.fixture()
def datamodule(data_config):
    return BenchmarkDataModule(
        data_config, num_workers=1, batch_size=2, pin_memory=False
    )


def test_dca_fit(model, datamodule, tmp_path: Path):
    """Run fit to test the DCA model."""

    trainer = Trainer(max_epochs=1, default_root_dir=tmp_path, log_every_n_steps=1)
    trainer.fit(model, datamodule)