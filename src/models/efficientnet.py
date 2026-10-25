from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_efficientnet(num_classes, pretrained=True):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    model.classifier[1] = __import__('torch').nn.Linear(model.classifier[1].in_features, num_classes)
    return model
