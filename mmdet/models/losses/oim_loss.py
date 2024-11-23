# This file is part of COAT, and is distributed under the
# OSI-approved BSD 3-Clause License. See top-level LICENSE file or
# https://github.com/Kitware/COAT/blob/master/LICENSE for details.

import torch
import torch.nn.functional as F
from torch import autograd, nn

from mmdet.registry import MODELS
from mmengine.dist import all_gather_object, get_data_device, cast_data_device

class OIM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, lut, cq, header, momentum):
        outputs_labeled = inputs.mm(lut.t())
        outputs_unlabeled = inputs.mm(cq.t())
        # print(inputs)
        # print(targets)
        input_device = get_data_device(inputs)
        
        inputs = torch.cat(cast_data_device(all_gather_object(inputs), input_device), dim=0)
        targets = torch.cat(cast_data_device(all_gather_object(targets), input_device), dim=0)
        # print(targets)
        ctx.save_for_backward(inputs, targets, lut, cq, header, momentum)
        
        return torch.cat([outputs_labeled, outputs_unlabeled], dim=1)

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, lut, cq, header, momentum = ctx.saved_tensors

        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(torch.cat([lut, cq], dim=0))
            if grad_inputs.dtype == torch.float16:
                grad_inputs = grad_inputs.to(torch.float32)

        for x, y in zip(inputs, targets):
            if y < len(lut):
                lut[y] = momentum * lut[y] + (1.0 - momentum) * x
                lut[y] /= lut[y].norm()
            else:
                cq[header] = x
                header = (header + 1) % cq.size(0)
        return grad_inputs, None, None, None, None, None


def oim(inputs, targets, lut, cq, header, momentum=0.5):
    return OIM.apply(inputs, targets, lut, cq, torch.tensor(header), torch.tensor(momentum))


@MODELS.register_module()
class OIMLoss(nn.Module):
    def __init__(self, num_features, num_pids, num_cq_size, oim_momentum, oim_scalar, loss_weight):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar
        self.loss_weight = loss_weight

        self.register_buffer("lut", torch.zeros(self.num_pids, self.num_features))
        self.register_buffer("cq", torch.zeros(self.num_unlabeled, self.num_features))

        self.header_cq = 0

    def forward(self, inputs, roi_label):
        # merge into one batch, background label = 0
        # targets = torch.cat(roi_label)
        label = roi_label - 1  # background label = -1

        inds = label >= 0
        
        label = label[inds]
        inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)

        projected = oim(inputs, label, self.lut, self.cq, self.header_cq, momentum=self.momentum)
        # projected - Tensor [M, lut+cq], e.g., [M, 482+500]=[M, 982]
        
        projected *= self.oim_scalar        

        self.header_cq = (
            self.header_cq + (label >= self.num_pids).long().sum().item()
        ) % self.num_unlabeled
        
        if (label == 5554).all():
            loss_oim = inputs.sum() * 0
        else:
            loss_oim = F.cross_entropy(projected, label, ignore_index=5554) * self.loss_weight
        
        return loss_oim * self.loss_weight
