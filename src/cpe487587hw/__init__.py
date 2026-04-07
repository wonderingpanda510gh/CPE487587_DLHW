from cpe487587hw._core import hello_from_bin
from .deepl import binary_classification, SimpleNN, ClassTrainer, ConvLayer, ImageNetCNN, CNNTrainer, ACCCruiseDataset, ACCNet, ACCTrainer, DiceLoss, NormalizationStats, GenModelTrainer, metrics
from .animation import animate_weight_heatmap, animate_large_heatmap



def hello() -> str:
    return hello_from_bin()
