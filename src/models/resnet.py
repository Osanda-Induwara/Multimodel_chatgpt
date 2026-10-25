from torchvision.models import ResNet18_Weights, resnet18


def build_resnet(num_classes, pretrained=True):
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = __import__('torch').nn.Linear(model.fc.in_features, num_classes)
    return model
