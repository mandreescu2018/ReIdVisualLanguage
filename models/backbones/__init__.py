from .resnet import ResNet50Backbone

BACKBONES = {
    "resnet50": ResNet50Backbone,
    # "resnet101":  ResNet101Backbone,
    # "osnet":      OSNetBackbone,
    # "vit_small":  ViTSmallBackbone,
}