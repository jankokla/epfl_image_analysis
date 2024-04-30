"""models.py: helper functions to get certain models."""
import timm

NUM_CLS = 16


def get_model(model: str) -> timm.models:
    """
    Some models use 'classifier' as their last layer, some others 'fc'.
    Thus, this function offers common interface for:
        1. getting model from timm;
        2. freezing all the layers except classification head.

    Args:
        model (str): name used in HuggingFace

    Returns:
        model (timm.models): model object with unfrozen classification head
    """

    decision_dict = {
        'efficientnet_b3.ra2_in1k': _get_classifier_head_model,
        'inception_v3.tv_in1k': _get_fc_head_model,
        'resnet50.a1_in1k': _get_fc_head_model
    }

    return decision_dict[model](model)


def _get_classifier_head_model(model_name: str) -> timm.models:
    """Return efficientnet pretrained version from timm."""
    model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLS)
    model = freeze_layers(model)

    # unfreeze the last layer
    model.classifier.requires_grad = True
    model.classifier.bias.requires_grad = True

    return model


def _get_fc_head_model(model_name: str) -> timm.models:
    """Return inception pretrained version from timm."""
    model = timm.create_model(model_name, pretrained=True,num_classes=NUM_CLS)
    model = freeze_layers(model)

    # unfreeze the last layer
    model.fc.requires_grad = True
    model.fc.bias.requires_grad = True

    return model


def freeze_layers(model: timm.models) -> timm.models:
    """Freeze all layers of the model as we don't want to retrain encoder."""
    for param in model.parameters():
        param.requires_grad = False

    return model
