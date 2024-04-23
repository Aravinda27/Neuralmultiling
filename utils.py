import torch
import dateutil.tz
import time
import logging
import os


import numpy as np
from sklearn.metrics import roc_curve, det_curve
from datetime import datetime
import matplotlib.pyplot as plt
from collections import namedtuple

plt.switch_backend('agg')

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_pretrained_weights(model, checkpoint):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    checkpoint_file = torch.load(checkpoint)
    pretrain_dict = checkpoint_file['state_dict']
    model_dict = model.state_dict()
    pretrain_dict = {
        k: v
        for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    # meters:  [batch_time, losses, top1.val, top5.val] OR [batch_time, data_time, losses, top1.val, top5.val, alpha_entropies] OR [batch_time]
    def __init__(self, num_batches, *meters, prefix="", logger=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)       #Like [{:2d/20}]
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
         
        entries += [str(meter) for meter in self.meters]
        if self.logger:
            self.logger.info('\t'.join(entries))
        else:
            print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def compute_eer(distances, labels):
    fprs, tprs, _ = roc_curve(labels, distances)
    fmr, fnmr, _ = det_curve(labels, distances)
    eer = fprs[np.nanargmin(np.absolute((1 - tprs) - fprs))]	
    return fprs, tprs, fmr, fnmr, eer


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)			#max((1,)): 1 and max((1,6)): 6
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)		#see the examples of topk. Output is of (256, 1251) and target is of (256). And we need to find the max value along each row, so topk(x, 1) will give 1 largest value along each row. topk(x,2) will give 2 largest element along each row. 3rd Argument is for largest, 4th Argument is for returning in sorted order
        # pred will be containing the maxk number of indices along each row(because of 2nd parameter). As we need to find top5 predictions. pred will be of [256, 5]  
      
        pred = pred.t()
       
        #view(1,-1) will convert 1D tensor(of size 256) to 2D tensor(of size 1, 256). Golden output or Target for topk should be like [[9,9,9,9,9], [14,14,14,14,14],..].T. so to transform it into like we are using expand_as
        correct = pred.eq(target.view(1, -1).expand_as(pred))   #Both pred and target should be of same dim. And .eq returns a same dim tensor with elements as pred[i][j] == target[i][j]
        
        res = []
        #This will find the top1 and topk accuracies and return it
        for k in topk:
            #correct is of dim (5,256). correct[:k] will give k rows with all columns. so for topK: (1,5), First top predicted will be taken and second time top k predicted will be taken
            correct_k = torch.reshape(correct[:k],(1,-1)).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

#log_dir: 'logs/resnet18_veri_(current DateTime)/Log' ------ phase: 'train'
def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')          # time_str: '2022-07-25-12-27'
    log_file = '{}_{}.log'.format(time_str, phase)          # log_file: 'train_2022-07-25-12-27'
    final_log_file = os.path.join(log_dir, log_file)            #final_log_file: 'logs/resnet18_veri_(current DateTime)/Log/train_2022-07-25-12-27'
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger

#root_dir: 'logs'---exp_name: 'resnet18_veri'
def set_path(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(root_dir, exp_name)     #exp_path: 'logs/resnet18_veri'
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp             #prefix: 'logs/resnet18_veri_(current DateTime)'
    os.makedirs(prefix)
    path_dict['prefix'] = prefix                        #path_dict['prefix']: string representing the model location

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')           #'logs/resnet18_veri_(current DateTime)/Model'
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path                  #path_dict['ckpt_path']: string representing the checkpointing location

    log_path = os.path.join(prefix, 'Log')              #'logs/resnet18_veri_(current DateTime)/Log'
    os.makedirs(log_path)
    path_dict['log_path'] = log_path                    #path_dict['log_path']: string representing the location of the log file

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples')       #sample_path: 'logs/resnet18_veri_(current DateTime)/Samples'
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    return path_dict


def to_item(x):
    """Converts x, possibly scalar and possibly tensor, to a Python scalar."""
    if isinstance(x, (float, int)):
        return x

    if float(torch.__version__[0:3]) < 0.4:
        assert (x.dim() == 1) and (len(x) == 1)
        return x[0]

    return x.item()


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))

def drop_path(x, drop_prob):
  #print(x.size())
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    try:
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)         #input size is of shape [a,b,c,d] (a is x.size(0)]), mask is randomly generated tensor of shape (a,1,1,1) whose element can be 1 or 0 based on the probability 'keep_prob' which is based on bernouli distribution
        # ex: [[[[0]]], [[[1]]], [[[1]]], [[[0]]]]
    except AttributeError:
        print(x)
        mask = torch.cuda.FloatTensor(1, 1, 1, 1).bernoulli_(keep_prob)
        
    x.div_(keep_prob)           # Each element of the x will be divided by keep_prob 
    x.mul_(mask)                # Each element of the x will be multiplied by mask. say, mask is [[[[0]]], [[[1]]], [[[1]]], [[[0]]]] then all elements of x[0] and x[3] will become 0.
  return x


def gumbel_softmax(logits, tau=1, hard=True, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor
    """
    Samples from the `Gumbel-Softmax distribution`_ and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Gumbel-Softmax distribution:
        https://arxiv.org/abs/1611.00712
        https://arxiv.org/abs/1611.01144
    """
    def _gen_gumbels():
        gumbels = -torch.empty_like(logits).exponential_().log()        
        if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():        #if any of the element in the tensor gumbel is nan or inf, then 
            # to avoid zero in exp output
            gumbels = _gen_gumbels()
        return gumbels

    gumbels = _gen_gumbels()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft

    """ if torch.isnan(ret).sum():
        import ipdb
        ipdb.set_trace()
        raise OverflowError(f'gumbel softmax output: {ret}')"""
    return ret
