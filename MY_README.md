## Basic info
Refer [README.md](README.md).

## Change Log


> Aug 29, 2025
- 修改 [dca_wrapper.py](src/foundation_models/dca_wrapper.py) ，`DCASegmentation.loss()` 新增 `affinity matrix l2 loss`和`random projection l2 loss`
- 新增 [visualize/plot.py](visualize/plot.py)，`pca_with_upsample()` 使用bilinear插值和PCA降维得到可视化特征图


> Aug 23, 2025

**训练文件**: [src/my_train.py](src/my_train.py)

**配置文件**:
- 主文件: [src/configs/config.yaml](src/configs/config.yaml)
- 次级模型配置文件: [src/configs/model/dca_seg.yaml](src/configs/model/dca_seg.yaml)
- 次级数据集配置文件: [src/configs/dataset/geobench_cashew.yaml](src/configs/dataset/geobench_cashew.yaml) (原始数据有13通道，但DOFA的作者在这里只使用了9通道；而且缺少band_wavelengths项，波长列表是从原始数据中读取手动添加进来的)

**模型文件**：
- 模型训练Class: [src/foundation_models/dca_wrapper.py](src/foundation_models/dca_wrapper.py)
- 模型定义Class: [src/foundation_models/DCA/dca_seg.py](src/foundation_models/DCA/dca_seg.py)
- DOFA的Patch_Embedding定义Class: [src/foundation_models/DCA/wave_dynamic_layer.py](src/foundation_models/DCA/wave_dynamic_layer.py)
