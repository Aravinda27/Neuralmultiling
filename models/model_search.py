from collections import namedtuple
import math
from turtle import shape
from torch import nn
import torch.nn.functional as F
from operations import *
from utils import Genotype
from utils import gumbel_softmax, drop_path


class MixedOp(nn.Module):

    def __init__(self, C, stride, PRIMITIVES):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.tempC = C
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False)).cuda()
            self._ops.append(op)

    def forward(self, x, weights):
        """
        This is a forward function.
        :param x: Feature map
        :param weights: A tensor of weight controlling the path flow
        :return: A weighted sum of several path
        """
        output = 0
        for op_idx, op in enumerate(self._ops):
            if weights[op_idx].item() != 0:
                if math.isnan(weights[op_idx]):
                    raise OverflowError(f'weight: {weights}')
            output += weights[op_idx] * op(x)
        return output


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, layer_index):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.layers_index = layer_index
        self.primitives = self.PRIMITIVES['primitives_reduct' if reduction else 'primitives_normal']   #It is List of List: [[8 ops], [8 ops], ..(14 times total)]

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False).cuda()        #If input shape is [4,48,R,C], then output shape is [4, 48, Math.ceil((R - 1)/2) + 1, Math.ceil((C - 1)/2) + 1] 
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False).cuda()         #If input shape is [4,48,R,C], then output shape is [4, 48, R, C ] 
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False).cuda()
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()

        edge_index = 0

        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, self.primitives[edge_index])
                self._ops.append(op)
                edge_index += 1
        """
        Connection Details in a Cell
        i = 0:
            j = 0: x0 to x2
            j = 1: x1 to x2
        i = 1
            j = 0: x0 to x3
            j = 1: x1 to x3
            j = 2: x2 to x3
        i = 2
            j = 0: x0 to x4
            j = 1: x1 to x4
            j = 2: x2 to x4
            j = 3: x3 to x4
        i = 3
            j = 0: x0 to x5
            j = 1: x1 to x5
            j = 2: x2 to x5
            j = 3: x3 to x5
            j = 4: x4 to x5
        """
    def forward(self, s0, s1, weights, drop_prob=0.0):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):

            if drop_prob > 0. and self.training:
                s = sum(drop_path(self._ops[offset + j](h, weights[offset + j]), drop_prob) for j, h in enumerate(states))      
            else:
                s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, primitives,
                 steps=4, multiplier=4, stem_multiplier=3, drop_path_prob=0.0):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.drop_path_prob = drop_path_prob

        nn.Module.PRIMITIVES = primitives	  #Adding a variable(Here, orderedDict) in nn.module which is being later used in Cell class. 

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(1, C_curr, 3, padding=1, bias=False),     # input size will be [4,1,300,257], so op size will be [4, C_curr(or 48), 300, 257]
            nn.BatchNorm2d(C_curr),     #no shape change
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)        #op size will be [4, 48, 150, 129]
        ).cuda()

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        ##changed portion
        reduction_prev = False
        normal_layer_index = 0
        reduct_layer_index = 0
        layer_index = 0
        for i in range(self._layers):
            if i in [self._layers // 3, 2 * self._layers // 3]:       #Reduction layer will be only placed at layers // 3 and (2 * layers) // 3 position
                C_curr *= 2
                reduction = True
                layer_index = reduct_layer_index
                reduct_layer_index += 1
            else:
                reduction = False
                layer_index = normal_layer_index
                normal_layer_index += 1
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, layer_index)
            ##changed portion
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1)).cuda()
        self.classifier = nn.Linear(C_prev, self._num_classes).cuda()

        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._embed_dim, self._layers, self._criterion,
                            self.PRIMITIVES, drop_path_prob=self.drop_path_prob).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input, discrete=False):
        input = input.unsqueeze(1)                      #x.shape: [4,300,257], unsqueeze insert a new dimension at the specified dim. x.unsqueeze(1) will give a shape of [4,1,300,257]. This dim0 is the batch size
        s0 = s1 = self.stem(input)               #  op size: [4, 48, 150, 129]             
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                if discrete:
                    weights = self.alphas_reduce[cell.layers_index]
                else:
                    weights = gumbel_softmax(F.log_softmax(self.alphas_reduce[cell.layers_index], dim=-1))
            else:
                if discrete:
                    weights = self.alphas_normal[cell.layers_index]
                else:
                    weights = gumbel_softmax(F.log_softmax(self.alphas_normal[cell.layers_index], dim=-1))
            s0, s1 = s1, cell(s0, s1, weights, self.drop_path_prob)
        v = self.global_pooling(s1)
        v = v.view(v.size(0), -1)
        if not self.training:
            return v

        y = self.classifier(v)

        return y

    def forward_classifier(self, v):
        y = self.classifier(v)
        return y

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)
    ## Changed portion compared to autospeech

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(self.PRIMITIVES['primitives_normal'][0])

        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(self._layers - 2, k, num_ops, requires_grad=True, device="cuda")).cuda()           #Alpha parameter for the normal layer
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(2, k, num_ops, requires_grad=True, device="cuda")).cuda()            #Alpha parameter for the reduction layer
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def compute_arch_entropy(self, dim=-1):                 #Called from functions.py
        alpha = self.arch_parameters()[0]           #Normal layer Alpha parameter.
        prob = F.softmax(alpha, dim=dim)
        log_prob = F.log_softmax(alpha, dim=dim)
        entropy = - (log_prob * prob).sum(-1, keepdim=False)
        return entropy.mean(dim = -1)                            #entropy shape [no. of normal_layers, 14, 8], Taking mean for each cell (i.e. 1st dimension) will give output of shape [14, 8]
    #Changed portion layer_index
    def genotype(self):
        def _parse(weights, layer_index, normal=True):
            PRIMITIVES = self.PRIMITIVES['primitives_normal' if normal else 'primitives_reduct']
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[layer_index][start:end].copy()			# Rows of weight from start(inclusive) to end(exclusive)
                try:
                    edges = sorted(range(i + 2), key=lambda x: -max(
                        W[x][k] for k in range(len(W[x])) if k != PRIMITIVES[x].index('none')))[:2]     #for k = 0, no element will be added inside -max(..) as 'index' is at index0 in the primitive list
                except ValueError:  # This error happens when the 'none' op is not present in the ops
                    edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if 'none' in PRIMITIVES[j]:
                            if k != PRIMITIVES[j].index('none'):
                                if k_best is None or W[j][k] > W[j][k_best]:
                                    k_best = k
                        else:
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    # print('operation: ' + str(PRIMITIVES[start+j][k_best]) + ', j = ' + str(j))
                    # ----------------Operation --------Layer index------edge
                    gene.append((PRIMITIVES[start+j][k_best], j))
                start = end
                n += 1
            return gene
         ##Changed portion of the dataset
        
        gene_normal = {i : _parse(F.softmax(self.alphas_normal, dim = -1).data.cpu().numpy(), i, True) for i in range(self._layers - 2)}             # dim := -1 is specifying that softmax has to be taken along each row. Dimension wont change, just sum of all elements in each row will be adding to 1.
        gene_reduce = {i : _parse(F.softmax(self.alphas_reduce, dim = -1).data.cpu().numpy(), i, False) for i in range(2)}

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype

