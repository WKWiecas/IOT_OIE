import torch
import math
from torch import nn
from OT.my_utils import normalize
from OT.my_sinkhorn import wasserstein_kmeans, multihead_attn


class OTKernel(nn.Module):
    def __init__(self, in_dim,  heads=1, eps=0.1, max_iter=10,
                 log_domain=False, position_encoding=None, position_sigma=0.1):
        super().__init__()
        self.in_dim = in_dim
        # self.out_size = out_size
        self.heads = heads
        self.eps = eps
        self.max_iter = max_iter

        # self.weight = nn.Parameter(
        #     torch.Tensor(heads, out_size, in_dim))

        self.log_domain = log_domain
        self.position_encoding = position_encoding
        self.position_sigma = position_sigma

    #     self.reset_parameter()
    #
    # def reset_parameter(self):
    #     stdv = 1. / math.sqrt(self.out_size)
    #     for w in self.parameters():
    #         w.data.uniform_(-stdv, stdv)

    def get_position_filter(self, input, out_size):
        if input.ndim == 4:
            in_size1 = input.shape[1]
            in_size2 = input.shape[2]
            out_size = int(math.sqrt(out_size))
            if self.position_encoding is None:
                return self.position_encoding
            elif self.position_encoding == "gaussian":
                sigma = self.position_sigma
                a1 = torch.arange(1., in_size1 + 1.).view(-1, 1) / in_size1
                a2 = torch.arange(1., in_size2 + 1.).view(-1, 1) / in_size2
                b = torch.arange(1., out_size + 1.).view(1, -1) / out_size
                position_filter1 = torch.exp(-((a1 - b) / sigma) ** 2)
                position_filter2 = torch.exp(-((a2 - b) / sigma) ** 2)
                position_filter = position_filter1.view(
                    in_size1, 1, out_size, 1) * position_filter2.view(
                    1, in_size2, 1, out_size)
            # if self.weight.is_cuda:
            #     position_filter = position_filter.cuda()
            return position_filter.reshape(1, 1, in_size1 * in_size2, out_size * out_size)
        in_size = input.shape[1]
        if self.position_encoding is None:
            return self.position_encoding
        elif self.position_encoding == "gaussian":
            # sigma = 1. / out_size
            sigma = self.position_sigma
            a = torch.arange(0., in_size).view(-1, 1) / in_size
            b = torch.arange(0., out_size).view(1, -1) / out_size
            position_filter = torch.exp(-((a - b) / sigma) ** 2)
        elif self.position_encoding == "hard":
            # sigma = 1. / out_size
            sigma = self.position_sigma
            a = torch.arange(0., in_size).view(-1, 1) / in_size
            b = torch.arange(0., out_size).view(1, -1) / out_size
            position_filter = torch.abs(a - b) < sigma
            position_filter = position_filter.float()
        else:
            raise ValueError("Unrecognizied position encoding")
        # if self.weight.is_cuda:
        #     position_filter = position_filter.cuda()
        position_filter = position_filter.view(1, 1, in_size, out_size)
        return position_filter

    def get_attn(self, input,target, mask=None, position_filter=None, iou_cost=None):
        """Compute the attention weight using Sinkhorn OT
        input: batch_size x in_size x in_dim
        mask: batch_size x in_size
        self.weight: heads x out_size x in_dim
        output: batch_size x (out_size x heads) x in_size
        """
        return multihead_attn(
            input, target, mask=mask, eps=self.eps,
            max_iter=self.max_iter, log_domain=self.log_domain,
            position_filter=position_filter, iou_cost=iou_cost)

    def forward(self, input,target, iou_cost, mask=None):
        """
        input: batch_size x in_size x in_dim
        output: batch_size x out_size x (heads x in_dim)
        """
        self.out_size = target.size(1)
        batch_size = input.shape[0]
        position_filter = self.get_position_filter(input, self.out_size)
        in_ndim = input.ndim
        if in_ndim == 4:
            input = input.view(batch_size, -1, self.in_dim)
        # attn_weight: batch_size x out_size x heads x in_size
        attn_weight = self.get_attn(input,target, mask, position_filter, iou_cost)
        
        # attn_weight: batch_size x out_size x in_size
        attn_weight = attn_weight.view(batch_size, self.out_size * self.heads, -1)
        return attn_weight
        # output = torch.bmm(
        #     attn_weight.view(batch_size, self.out_size * self.heads, -1), input)
        # if in_ndim == 4:
        #     out_size = int(math.sqrt(self.out_size))
        #     output = output.reshape(batch_size, out_size, out_size, -1)
        # else:
        #     output = output.reshape(batch_size, self.out_size, -1)
        # return output

    def unsup_train(self, input, wb=False, inplace=True, use_cuda=False):
        """K-meeans for learning parameters
        input: n_samples x in_size x in_dim
        weight: heads x out_size x in_dim
        """
        input_normalized = normalize(input, inplace=inplace)
        block_size = int(1e9) // (input.shape[1] * input.shape[2] * 4)
        print("Starting Wasserstein K-means")
        weight = wasserstein_kmeans(
            input_normalized, self.heads, self.out_size, eps=self.eps,
            block_size=block_size, wb=wb, log_domain=self.log_domain, use_cuda=use_cuda)
        self.weight.data.copy_(weight)

    # def random_sample(self, input):
    #     idx = torch.randint(0, input.shape[0], (1,))
    #     self.weight.data.copy_(input[idx].view_as(self.weight))