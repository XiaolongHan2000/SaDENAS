import random

from chromosomes import *
import torch

class Population:
	def __init__(self, pop_size, steps, device = torch.device("cpu")):
		self._pop_size = pop_size
		self._device = device
		self._steps = steps
		self.population = []
		for _ in range(pop_size):
			self.population.append(chromosome(steps, self._device))

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
			self.population.append(chromosome(self._steps, self._device))

	def get_population_top50(self):
		# for i in range(50):
		# 	if self.population[i] > self.population[i+50]:
		# 		self.population[i] = self.population[i]
		# 	else:
		# 		self.population[i] = self.population[i+50]
		self.population = self.population[:50]
		return self.population

	def pop_pop(self, indices_to_pop):
		for index in sorted(indices_to_pop, reverse=True):
			self.population.pop(index)
		return self.population
