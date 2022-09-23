# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------

Copyright (C) 2019 Université catholique de Louvain (UCLouvain), Belgium.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

------------------------------------------------------------------------------

 "function.py" - Functional definition of the TrainingHook class (module.py).
 
 Project: DRTP - Direct Random Target Projection

 Authors:  C. Frenkel and M. Lefebvre, Université catholique de Louvain (UCLouvain), 09/2019

 Cite/paper: C. Frenkel, M. Lefebvre and D. Bol, "Learning without feedback: Direct random target projection
    as a feedback-alignment algorithm with layerwise feedforward training," arXiv preprint arXiv:1909.01311, 2019.

------------------------------------------------------------------------------
"""


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
                grad_output_est = labels.mm(fixed_fb_weights.view(-1,prod(fixed_fb_weights.shape[1:]))).view(grad_output.shape) # * grad_output
                input_ = input.detach()
            else:
                factor = abs(input.detach() - input_)#/abs(input_)
                factor = torch.clamp(factor, 0, 2)
                grad_output_est = labels.mm(fixed_fb_weights.view(-1,prod(fixed_fb_weights.shape[1:]))).view(grad_output.shape) # * factor # * grad_output
                input_ = input.detach()
            grad_output_est = labels.mm(fixed_fb_weights.view(-1,prod(fixed_fb_weights.shape[1:]))).view(grad_output.shape)
        elif train_mode == "BRP":
            with torch.no_grad():
                LB = labels.mm(fixed_fb_weights.view(-1,prod(fixed_fb_weights.shape[1:]))).view(grad_output.shape)
                input_ = input.clone() * utils.args.brpscale
                grad_output_est = LB - input_
        else:
            raise NameError("=== ERROR: training mode " + str(train_mode) + " not supported")

        return grad_output_est, None, None, None, None, None

trainingHook = HookFunction.apply