from operations import *
from utils import drop_path
from torchsummary import summary


class Cell(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    print(C_prev_prev, C_prev, C)

    #----------------------------------------------------
    #----------Initial Preprocessing Step ----------------
    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
    #-----------------------------------------------------
    #-----------------------------------------------------

    if reduction:
      op_names, indices = zip(*genotype.reduce)              #getting operation name and edge index of reduction cell at index(layer index)
      concat = genotype.reduce_concat
    else:
      op_names, indices = zip(*genotype.normal)                #getting operation name and edge index of normal cell at index(layer index)
      concat = genotype.normal_concat
    self._compile(C, op_names, indices, concat, reduction)

  def _compile(self, C, op_names, indices, concat, reduction):
    assert len(op_names) == len(indices)                          #No. of edges and operations performed should be same, otherwise something is fishy
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 2 if reduction and index < 2 else 1
      op = OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, s0, s1, drop_prob):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):                  #i = 0, (0, 1) will be accessed; i = 1, (2, 3) will be accessed, ...
      h1 = states[self._indices[2 * i]]           
      h2 = states[self._indices[2 * i + 1]]
      op1 = self._ops[2 * i]
      op2 = self._ops[2 * i + 1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, genotype):
    super(Network, self).__init__()
    self._C = C                           #No. of channels
    self._num_classes = num_classes         #no. of classes
    self._layers = layers                     #no. of layers

    self.stem0 = nn.Sequential(                                                 #input is [48, 1, 300 ,257]
      nn.Conv2d(1, C // 2, kernel_size=3, stride=2, padding=1, bias=False),       #output size is [48, 8, 150, 129]
      nn.BatchNorm2d(C // 2),                                                         #output size won't change in BatchNorm2d and ReLU
      nn.ReLU(inplace=True),                                                            
      nn.Conv2d(C // 2, C, kernel_size=3, stride=2, padding=1, bias=False),             #output size is [48, 16, 75, 65]
      nn.BatchNorm2d(C),
    )
    '''self.stem01 = nn.Conv2d(1, C // 2, kernel_size=3, stride=2, padding=1, bias=False)   
    self.stem02 = nn.BatchNorm2d(C // 2)
    self.stem03 = nn.ReLU(inplace=True)
    self.stem04 = nn.Conv2d(C // 2, C, kernel_size=3, stride=2, padding=1, bias=False)
    self.stem05 = nn.BatchNorm2d(C)'''


    self.stem1 = nn.Sequential(           #input size is [48, 16, 75, 65]
      nn.ReLU(inplace=True),
      nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),          #output size is [48, 16, 38, 33]
      nn.BatchNorm2d(C),
    )

    C_prev_prev, C_prev, C_curr = C, C, C

    self.cells = nn.ModuleList()
    reduction_prev = True
    for i in range(layers):
      if i in [layers // 3, 2 * layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    input = input.unsqueeze(1)
    #print('input shape: ' + str(input.shape))
    assert not (torch.isnan(input).any().item() or torch.isinf(input).any().item())
    
    s0 = self.stem0(input)
    #----------------------------------------
    '''for param in self.stem01.parameters():
      #print('param shape ' + str(param.shape))
      assert not (torch.isnan(param).any().item())
    s0 = self.stem01(input)
    if torch.isinf(s0).any().item():
      print('s0 is -inf')
    if torch.isnan(s0).any().item():
      print('s0 is nan')
    assert not (torch.isnan(s0).any().item() or torch.isinf(s0).any().item())
    s0 = self.stem02(s0)
    assert not (torch.isnan(s0).any().item() or torch.isinf(s0).any().item())
    s0 = self.stem03(s0)
    assert not (torch.isnan(s0).any().item() or torch.isinf(s0).any().item())
    s0 = self.stem04(s0)
    assert not (torch.isnan(s0).any().item() or torch.isinf(s0).any().item())
    s0 = self.stem05(s0)
    assert not (torch.isnan(s0).any().item() or torch.isinf(s0).any().item())'''
    #----------------------------------------
    #assert not torch.isnan(s0).any().item()
    #print('s0 shape: ' + str(s0.shape))
    s1 = self.stem1(s0)
    #print('s1 shape: ' + str(s1.shape))
    assert not torch.isnan(s0).any().item()
    assert not torch.isnan(s1).any().item()
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      #print('cell: ' + str(i))
      #if i == 0:
        #print('s0 shape: ' + str(s0.shape) + ', s1 shape: ' + str(s1.shape))
      assert not torch.isnan(s0).any().item()
      assert not torch.isnan(s1).any().item()
    v = self.global_pooling(s1)
    v = v.view(v.size(0), -1)
    if not self.training:
      return v

    y = self.classifier(v)
    return y


  def forward_classifier(self, v):
    y = self.classifier(v)
    return y




