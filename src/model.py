"""Transfer learning model: pretrained backbone + new 7-class head.

Quick test (run from the project root):
    python src/model.py
"""
import timm
import torch

NUM_CLASSES = 7


def build_model(name="efficientnet_b0", num_classes=NUM_CLASSES, pretrained=True):
    """Load a model pretrained on ImageNet and swap its last layer for our 7 classes.

    pretrained=True  -> download the ImageNet-trained weights (the transferred knowledge)
    num_classes=7    -> timm removes the 1000-class ImageNet layer and adds a fresh 7-class one
    """
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def freeze_backbone(model):
    """Stage 1: lock the pretrained layers; only the new head can learn."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True


def unfreeze_all(model):
    """Stage 2: let every layer learn (fine-tuning, with a small learning rate)."""
    for p in model.parameters():
        p.requires_grad = True


def count_trainable(model):
    """How many weights will actually be updated during training."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)

    print("New head:", model.get_classifier())

    x = torch.randn(4, 3, 224, 224, device=device)  # fake batch of 4 images
    with torch.no_grad():
        out = model(x)
    print("Output shape:", tuple(out.shape), "-> 4 images x 7 class scores")

    print(f"Trainable weights (everything): {count_trainable(model):,}")
    freeze_backbone(model)
    print(f"Trainable weights (frozen):     {count_trainable(model):,}")
    unfreeze_all(model)
    print(f"Trainable weights (unfrozen):   {count_trainable(model):,}")