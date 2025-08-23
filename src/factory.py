"""Factory utily functions to create datasets and models."""

from src.foundation_models import (
    CromaModel,
    ScaleMAEModel,
    GFMModel,
    DinoV2Model,
    SoftConModel,
    DofaModel,
    SatMAEModel,
    AnySatModel,
    SenPaMAEModel,
    DCAModel,
)
from src.datasets.geobench_wrapper import GeoBenchDataset
from src.datasets.resisc_wrapper import Resics45Dataset
from src.datasets.benv2_wrapper import BenV2Dataset
from src.datasets.digital_typhoon_wrapper import DigitalTyphoonDataset
from src.datasets.tropical_cyclone_wrapper import TropicalCycloneDataset

from src.datasets.dummy_dataset import DummyWrapper

model_registry = {
    "croma": CromaModel,
    # "panopticon": PanopticonModel,
    "scalemae": ScaleMAEModel,
    "gfm": GFMModel,
    "dinov2": DinoV2Model,
    "softcon": SoftConModel,
    "dofa": DofaModel,
    "satmae": SatMAEModel,
    "anysat": AnySatModel,
    "senpamae": SenPaMAEModel,
    # Add other model mappings here
    "dca": DCAModel
}

dataset_registry = {
    "geobench": GeoBenchDataset,
    "resisc45": Resics45Dataset,
    "benv2": BenV2Dataset,
    "digital_typhoon": DigitalTyphoonDataset,
    "tropical_cyclone": TropicalCycloneDataset,
    # Add other dataset mappings here
    "dummy": DummyWrapper,
}


def create_dataset(config_data):
    dataset_type = config_data.dataset_type
    dataset_class = dataset_registry.get(dataset_type)
    if dataset_class is None:
        raise ValueError(f"Dataset type '{dataset_type}' not found.")
    dataset = dataset_class(config_data)
    # return the train, val, and test dataset
    return dataset.create_dataset()


def create_model(args, config_model, dataset_config=None):
    model_name = config_model.model_type
    model_class = model_registry.get(model_name)
    if model_class is None:
        raise ValueError(f"Model type '{model_name}' not found.")

    model = model_class(args, config_model, dataset_config)

    return model
