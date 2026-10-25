from .efficientnet import build_efficientnet
from .mobilenet import build_mobilenet
from .resnet import build_resnet
from .shufflenet import build_shufflenet
from .squeezenet import build_squeezenet

BUILDERS = {
    "efficientnet": build_efficientnet, "mobilenet": build_mobilenet,
    "resnet": build_resnet, "shufflenet": build_shufflenet, "squeezenet": build_squeezenet,
}


def create_model(name: str, num_classes: int, pretrained: bool = True):
    if name not in BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(BUILDERS)}")
    return BUILDERS[name](num_classes, pretrained)
