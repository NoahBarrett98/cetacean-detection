from torch import optim


def SGD(model, lr, momentum=0.9):
    return optim.SGD(
        [
            {"params": model.parameters(), "lr": lr}
        ],
        momentum=momentum,
    )


def Adam(model, lr, weight_decay=0.9):
    return optim.Adam(
        [
            {"params": model.base.parameters(), "lr": lr},
            {"params": model.classifier.parameters(), "lr": lr},
        ],
        weight_decay=weight_decay,
    )