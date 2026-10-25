from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def build_mobilenet(num_classes, pretrained=True):
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    model.classifier[3] = __import__('torch').nn.Linear(model.classifier[3].in_features, num_classes)
    return model
