import torch

from cell_operationsNAS201 import NAS_BENCH_201
from chromosomesNAS201 import *

class Population:
  def __init__(self, pop_size, num_edges, device = torch.device("cpu")):
    self._pop_size = pop_size
    self._device = device
    self._num_edges = num_edges
    self.population = []
    for _ in range(pop_size):
      self.population.append(chromosome(self._num_edges, self._device, NAS_BENCH_201))

  def get_population_size(self):
    return len(self.population)

  def get_population(self):
    return self.population

  def print_population(self):
    for p in self.population:
      print(p.get_fitness())

  def pop_sort(self):
    self.population.sort(key = lambda x: x.get_fitness(), reverse = True)

  def random_pop(self):
    self.population = []
    for _ in range(self._pop_size):
      self.population.append(chromosome(self._num_edges, self._device, NAS_BENCH_201))

  def pop_pop(self, indices_to_pop):
    for index in sorted(indices_to_pop, reverse=True):
      self.population.pop(index)
    return self.population
