import numpy as np
import random
from cell_operationsNAS201 import NAS_BENCH_201
from chromosomesNAS201 import *
from populationNAS201 import *

class DifferentialEvolution:
  def __init__(self, pop_size, tournament_size, device, mrate = 0.05):
    self._device = device
    self._pop_size = pop_size
    self._tournament_size = tournament_size
    self.mutation_rate = mrate

  def evolve(self, population):
    new_pop = Population(0, 0, self._device)
    for i in range(self._pop_size):
      new_pop.get_population().append(population.get_population()[i])
    for i in range(population.get_population_size()):
      candidates =np.random.choice(population.get_population_size(), size=3, replace=False)
      a = population.get_population()[candidates[0]]
      b = population.get_population()[candidates[1]]
      c = population.get_population()[candidates[2]]
      mutant = chromosome(population.get_population()[i]._num_edges, population.get_population()[i]._device, NAS_BENCH_201)
      mutant_factor = population.get_population()[i].get_mutate_factor() + (1-population.get_population()[i].get_mutate_factor())*(population.get_population()[0].get_mutate_factor()-population.get_population()[i].get_mutate_factor()) + population.get_population()[i].get_mutate_factor()*(b.get_mutate_factor()-c.get_mutate_factor())
      if mutant_factor > 0.9:
        mutant_factor = 0.9
      if mutant_factor < 0.1:
        mutant_factor = 0.1
      for chrom1, chrom2, chrom3, chrom4, chrom5, chrom6 in zip(population.get_population()[0].arch_parameters, a.arch_parameters, b.arch_parameters, c.arch_parameters, population.get_population()[i].arch_parameters, mutant.arch_parameters):
        for j in range(chrom1.shape[0]):
          chrom6[j].data.copy_(chrom5[j] + (1-mutant_factor)*(chrom1[j]-chrom5[j]) + mutant_factor*(chrom3[j]-chrom4[j]))
        mutant.update()
      cross_chrom = chromosome(population.get_population()[i]._num_edges, population.get_population()[i]._device, NAS_BENCH_201)
      for chrom1, chrom2, chrom3 in zip(mutant.arch_parameters, population.get_population()[i].arch_parameters, cross_chrom.arch_parameters):
        rand_j = np.random.randint(0, chrom1.shape[0])
        for j in range(chrom1.shape[0]):
          if np.random.rand() >= 0.5 or j == rand_j:
            chrom3[j].data.copy_(chrom1[j].data)
          else:
            chrom3[j].data.copy_(chrom2[j].data)
          cross_chrom.update()
      cross_chrom.set_mutate_factor(mutant_factor)
      new_pop.get_population().append(cross_chrom)
    return new_pop

  @staticmethod
  def verify_crossover(x, y, z):
    for c1, c2, c3 in zip(x.arch_parameters, y.arch_parameters, z.arch_parameters):
      for i in range(c1.shape[0]):
        if torch.all(c1[i].eq(c3[i])):
          print("{}: from 1st chromosome".format(i)) 
        elif(torch.all(c2[i].eq(c3[i]))):
          print("{}: from 2nd chromosome".format(i))

  @staticmethod
  def eq_chromosomes(x, y):
    for c1, c2 in zip(x.arch_parameters, y.arch_parameters):
      if torch.all(c1.eq(c2)) != True:
        return False
    return True












