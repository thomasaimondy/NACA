# -*- coding: utf-8 -*-

import torch
from torch.autograd import Function
from numpy import prod
import utils

class HookFunction(Function):
    @staticmethod
    def forward(ctx, input, labels, y, fixed_fb_weights, input_, train_mode):
        if train_mode in ["DFA", "sDFA", "DRTP", "BRP"]:
            ctx.save_for_backward(input, labels, y, fixed_fb_weights, input_)
        ctx.in1 = train_mode
        return input

    @staticmethod
    def backward(ctx, grad_output):
        train_mode          = ctx.in1
        if train_mode == "BP":
            return grad_output, None, None, None, None, None
        elif train_mode == "shallow":
            grad_output.data.zero_()
            return grad_output, None, None, None, None, None
        
        input, labels, y, fixed_fb_weights, input_ = ctx.saved_variables
        if train_mode == "DFA":
            grad_output_est = (y-labels).mm(fixed_fb_weights.view(-1,prod(fixed_fb_weights.shape[1:]))).view(grad_output.shape)
        elif train_mode == "sDFA":
            grad_output_est = torch.sign(y-labels).mm(fixed_fb_weights.view(-1,prod(fixed_fb_weights.shape[1:]))).view(grad_output.shape)
        elif train_mode == "DRTP":
            if input_ is None:
                grad_output_est = labels.mm(fixed_fb_weights.view(-1,prod(fixed_fb_weights.shape[1:]))).view(grad_output.shape)
                input_ = input.detach()
            else:
                factor = 2 * abs(input.detach() - input_)/abs(input_)
                factor = torch.clamp(factor, 0, 2)
                grad_output_est = labels.mm(fixed_fb_weights.view(-1,prod(fixed_fb_weights.shape[1:]))).view(grad_output.shape) 
                input_ = input.detach()

        elif train_mode == "BRP":
            LB = labels.mm(fixed_fb_weights.view(-1,prod(fixed_fb_weights.shape[1:]))).view(grad_output.shape)
            input_ = input * utils.args.brpscale
            grad_output_est = LB - input_
        else:
            raise NameError("=== ERROR: training mode " + str(train_mode) + " not supported")

        return grad_output_est, None, None, None, None, None

trainingHook = HookFunction.apply
