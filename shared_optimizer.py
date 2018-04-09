import math

import torch

# code from https://github.com/jingweiz/pytorch-rl


class SharedAdam(torch.optim.Adam):
    """Implements Adam algorithm with shared states.
    """
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = param.data.new().resize_as_(param.data).zero_()
                state['exp_avg_sq'] = param.data.new().resize_as_(param.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                state = self.state[param]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], param.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1**state['step'][0]
                bias_correction2 = 1 - beta2**state['step'][0]
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                param.data.addcdiv_(-step_size, exp_avg, denom)

        return loss