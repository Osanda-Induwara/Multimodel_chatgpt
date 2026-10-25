from torchvision.models import ShuffleNet_V2_X1_0_Weights, shufflenet_v2_x1_0


def build_shufflenet(num_classes, pretrained=True):
    model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None)
    model.fc = __import__('torch').nn.Linear(model.fc.in_features, num_classes)
    return model
