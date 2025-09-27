import torch
import torch.nn as nn
from torch.nn import functional as F


class OCCELoss(nn.Module):
    def __init__(self):
        super(OCCELoss, self).__init__()

    def forward(self, inputs, targets):
        N = inputs.shape[1]
        # multiply with N-1 for numerical stability, does not affect gradient
        ycomp = (N - 1) * F.softmax(-inputs, dim=1)
        y = torch.ones((targets.size(0), N), device=inputs.device)
        y.scatter_(1, targets.unsqueeze(1), 0.0)
        loss = - 1 / (N - 1) * torch.sum(y * torch.log(ycomp + 0.0000001), dim=1)

        return torch.mean(loss)


class CCELoss(nn.Module):
    def __init__(self, w_comp_entropy=1):
        super(CCELoss, self).__init__()
        self.w_comp = w_comp_entropy
        self.ce_criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        batch_size = len(targets)
        classes = inputs.shape[1]
        yhat = F.softmax(inputs, dim=1)
        yg = torch.gather(yhat, 1, torch.unsqueeze(targets, 1))
        yg_ = (1 - yg) + 1e-7  # avoiding numerical issues (first)
        px = yhat / yg_.view(len(yhat), 1)
        px_log = torch.log(px + 1e-10)  # avoiding numerical issues (second)
        y_zerohot = torch.ones(batch_size, classes).scatter_(
            1, targets.view(batch_size, 1).data.cpu(), 0)
        output = px * px_log * y_zerohot.to(yhat.device)
        loss_comp = torch.sum(output)
        loss_comp /= float(batch_size)
        loss_comp /= float(classes)

        loss_ce = self.ce_criterion(inputs, targets)

        loss = loss_ce + self.w_comp * loss_comp

        return loss


class SCLNLLoss(nn.Module):
    def __init__(self):
        super(SCLNLLoss, self).__init__()

    def forward(self, inputs, targets):
        y = torch.ones((targets.size(0), inputs.shape[1]), device=inputs.device)
        y.scatter_(1, targets.unsqueeze(1), 0.0)

        loss = - torch.sum(torch.log(1.0000001 - F.softmax(inputs, dim=1)) * y, dim=1)

        return torch.mean(loss)


class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, label_smoothing=0.1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        smoothing_value = self.label_smoothing / (num_classes - 1)
        soft_targets = torch.zeros_like(logits).fill_(smoothing_value)
        soft_targets.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing + smoothing_value)
        log_probs = F.log_softmax(logits, dim=1)

        loss = -torch.sum(soft_targets * log_probs, dim=1)

        return torch.mean(loss)



