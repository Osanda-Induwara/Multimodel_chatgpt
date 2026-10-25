from torchvision.models import SqueezeNet1_1_Weights, squeezenet1_1


def build_squeezenet(num_classes, pretrained=True):
    model = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT if pretrained else None)
    model.classifier[1] = __import__('torch').nn.Conv2d(512, num_classes, kernel_size=1)
    model.num_classes = num_classes
    return model
