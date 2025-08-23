from lightning import LightningDataModule
import torch
from src.factory import create_dataset


class BenchmarkDataModule(LightningDataModule):
    def __init__(self, dataset_config, batch_size, num_workers, pin_memory):
        super().__init__()
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        self.dataset_train, self.dataset_val, self.dataset_test = create_dataset(
            self.dataset_config
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )


if __name__ == "__main__":
    from types import SimpleNamespace
    
    
    # 使用 geobench_cashew.yaml 配置
    dataset_config = SimpleNamespace(
        dataset_type="geobench",
        benchmark_name="segmentation_v1.0",
        dataset_name="m-cashew-plant",
        task="segmentation",
        image_resolution=224,
        band_names=None,
        # band_wavelengths=[],
        # data_path=None,
        # multilabel=False,
        # additional_param=None,
        # ignore_index=None,
        # num_classes=7,
        # num_channels=9
    )
    
    # 创建数据模块实例
    data_module = BenchmarkDataModule(
        dataset_config=dataset_config,
        batch_size=4,
        num_workers=2,
        pin_memory=True
    )
    
    # 设置数据集
    print("Setting up datasets...")
    data_module.setup()
    
    # 获取数据加载器
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    
    print(f"Train dataset size: {len(data_module.dataset_train)}")
    print(f"Val dataset size: {len(data_module.dataset_val)}")
    print(f"Test dataset size: {len(data_module.dataset_test)}")
    
    # 测试加载一个批次
    print("Loading a sample batch...")
    for batch_idx, (images, masks) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Images shape: {images.shape}, Masks shape: {masks.shape}")
        break

