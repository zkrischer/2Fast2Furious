import torch

_steer_angles = {
    0: -0.5,
    1: -0.15,
    2: 0,
    3: 0.15,
    4: 0.5
}

def pred2steer(pred: torch.Tensor) -> float:
    probability_distribution = torch.nn.functional.softmax(pred)
    steer_class = torch.argmax(probability_distribution)
    angle = _steer_angles[steer_class.item()]
    return angle
