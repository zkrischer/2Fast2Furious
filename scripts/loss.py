
import torch
import torch.nn as nn

class EMDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.squared = True

    def forward(self, pred, target):
        # convert target to one-hot encoding
        if target.shape == torch.Size([1]):
            target = target.unsqueeze(0)
        target = torch.nn.functional.one_hot(target, num_classes=5).float().squeeze(0)
        # print(f'{pred=}')
        # print(f'{target=}')
        
        # Compute cumulative distributions along the ordered bins.
        cdf_pred = torch.cumsum(pred, dim=1)
        # print(f'{cdf_pred=}')
        cdf_target = torch.cumsum(target, dim=1)
        # print(f'{cdf_target=}')
        
        # Compute the difference between the cumulative distributions.
        cdf_diff = cdf_pred - cdf_target
        # print(f'{cdf_diff=}')
        
        if self.squared:
            # L2 distance: mean squared difference per sample, then square-root.
            emd = torch.mean(cdf_diff ** 2, dim=1)
            loss = torch.sqrt(emd)
        else:
            # L1 distance: mean absolute difference per sample.
            loss = torch.mean(torch.abs(cdf_diff), dim=1)
        # print(torch.mean(loss))
        # exit()
        return torch.mean(loss)
    
if __name__ == '__main__':
    gt = torch.tensor([[1]])
    # pred1 = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])
    # pred2 = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]])

    pred1 = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0]])
    pred2 = torch.tensor([[0.0, 0.5, 0.0, 0.0, 0.5]])

    crit = EMDLoss()
    print(crit(pred1, gt))
    print()
    print(crit(pred2, gt))