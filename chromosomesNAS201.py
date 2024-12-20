import torch
from genotypes import PRIMITIVES
from architect import _concat
import utils
import  random

class chromosome:
  def __init__(self, num_edges, device, search_space = None):
    self._device = device
    self._num_edges = num_edges
    
    assert search_space is not None, 'No search space is given'
    self.num_ops = len(search_space)

    self.alphas_normal = self.generate_parameters(self._num_edges).to(self._device)
    self.arch_parameters = [self.alphas_normal]
    
    self.objs = utils.AvgrageMeter()
    self.top1 = utils.AvgrageMeter()
    self.top5 = utils.AvgrageMeter()

    self.evaluated = False    
    self.tmp = []

    self.mutate_factor = random.uniform(0.1, 0.9)

  def accumulate(self):
    self.tmp.append(self.top1.avg)

  def update(self):
    self.alphas_normal = self.arch_parameters

  def set_fitness(self, value, top1, top5):
    self.objs.avg = value
    self.top1.avg = top1
    self.top5.avg = top5

  def get_len(self):
    #return self.k * len(self.arch_parameters)
    return self._num_edges

  def get_fitness(self):
    return self.top1.avg

  def get_all_metrics(self):
    return self.objs, self.top1, self.top5

  def get_arch_parameters(self):
    return self.arch_parameters

  def generate_parameters(self, k):
    return torch.rand(k, self.num_ops)
    #return torch.nn.functional.one_hot(torch.randint(low = 0, high = self.num_ops, size = (k, )), num_classes = self.num_ops)

  def get_mutate_factor(self):
    return self.mutate_factor

  def set_mutate_factor(self, mutate_factor):
    self.mutate_factor = mutate_factor