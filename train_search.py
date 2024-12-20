import os
import sys
import logging
import random
import torch.nn as nn
import genotypes
import utils
import torch.utils
import numpy as np
from torch.autograd import Variable
from model_search import Network
import argparse
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torch.utils
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
from population import *
import pickle
from ga import GeneticAlgorithm

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type = str, default = '/fasterdatasets/cifar-10', help = 'location of the data corpus')
parser.add_argument('--dir', type = str, default = None, help = 'location of trials')
parser.add_argument('--cutout', action = 'store_true', default = False, help = 'use cutout')
parser.add_argument('--cutout_length', type = int, default = 16, help = 'cutout length')
parser.add_argument('--batch_size', type = int, default = 64, help = 'batch size')
parser.add_argument('--epochs', type = int, default = 50, help = 'num of training epochs')
parser.add_argument('--seed', type = int, default = 0, help = 'random seed')
parser.add_argument('--gpu', type = int, default = 0, help = 'gpu device id')
parser.add_argument('--tsize', type = int, default = 10, help = 'Tournament size')
parser.add_argument('--num_elites', type = int, default = 50, help = 'Number of Elites')
parser.add_argument('--mutate_rate', type = float, default = 0.2, help = 'mutation rate')
parser.add_argument('--learning_rate', type = float, default = 0.025, help = 'init learning rate')
parser.add_argument('--learning_rate_min', type = float, default = 0.001, help = 'min learning rate')
parser.add_argument('--momentum', type = float, default = 0.9, help = 'momentum')
parser.add_argument('--weight_decay', type = float, default = 3e-4, help = 'weight decay')
parser.add_argument('--grad_clip', type = float, default = 5, help = 'gradient clipping')
parser.add_argument('--train_portion', type = float, default = 0.5, help = 'portion of training data')
parser.add_argument('--pop_size', type = int, default = 50, help = 'population size')
parser.add_argument('--report_freq', type = float, default = 50, help = 'report frequency')
parser.add_argument('--init_channels', type = int, default = 16, help = 'num of init channels')
parser.add_argument('--layers', type = int, default = 8, help = 'total number of layers')
parser.add_argument('--warm_up', type=int, default=0, help='warm up epochs')
args = parser.parse_args()

def warm_train(model, train_queue, criterion, optimizer):
	model.train()
	for step, (inputs, targets) in enumerate(train_queue):
		inputs = inputs.to(device)
		targets = targets.to(device)
		optimizer.zero_grad()
		logits = model(inputs)
		loss = criterion(logits, targets)
		loss.backward()
		nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
		optimizer.step()


# def train(model, train_queue, criterion, optimizer, gen):
# 	model.train()
# 	for i in range(len(population.get_population())):
# 		train_time = time.time()
# 		discrete_alphas = utils.discretize(population.get_population()[i].arch_parameters, device)
# 		model.copy_arch_parameters(discrete_alphas)
# 		for step, (inputs, targets) in enumerate(train_queue):
# 			n = inputs.size(0)
# 			inputs = inputs.to(device)
# 			targets = targets.to(device)
# 			optimizer.zero_grad()
# 			logits = model(inputs)
# 			loss = criterion(logits, targets)
# 			loss.backward()
# 			nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
# 			optimizer.step()
#
# 			prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
# 			population.get_population()[i].objs.update(loss.data, n)
# 			population.get_population()[i].top1.update(prec1.data, n)
# 			population.get_population()[i].top5.update(prec5.data, n)
#
# 			if (step + 1) % 100 == 0:
# 				logging.info("[{} Generation]".format(gen))
# 				logging.info("Using Training batch #{} for {}/{} architecture with loss: {}, prec1: {}, prec5: {}".format(step, i,
# 																								len(population.get_population()),
# 																								population.get_population()[i].objs.avg,
# 																								population.get_population()[i].top1.avg,
# 																								population.get_population()[i].top5.avg))

def train(model, train_queue, criterion, optimizer, gen):
	model.train()
	for step, (inputs, targets) in enumerate(train_queue):
		# discrete_alpha = utils.discretize(population.get_population()[step % args.pop_size].get_arch_parameters(), device)
		model.copy_arch_parameters(population.get_population()[step % args.pop_size].get_arch_parameters())
		# model.copy_arch_parameters(discrete_alpha)
		n = inputs.size(0)
		inputs = inputs.to(device)
		targets = targets.to(device)
		optimizer.zero_grad()
		logits = model(inputs)
		loss = criterion(logits, targets)
		loss.backward()
		nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
		optimizer.step()

		prec1, prec5 = utils.accuracy(logits, targets, topk = (1, 5))
		population.get_population()[step % args.pop_size].objs.update(loss.data, n)
		population.get_population()[step % args.pop_size].top1.update(prec1.data, n)
		population.get_population()[step % args.pop_size].top5.update(prec5.data, n)

		if (step + 1) % 100 == 0:
			logging.info("[{} Generation]".format(gen))
			logging.info("Using Training batch #{} for {}/{} architecture with loss: {}, prec1: {}, prec5: {}".format(step, step % args.pop_size,
																							len(population.get_population()),
																							population.get_population()[step % args.pop_size].objs.avg,
																							population.get_population()[step % args.pop_size].top1.avg,
																							population.get_population()[step % args.pop_size].top5.avg))

def validation(model, valid_queue, criterion, gen, population):
	model.eval()
	# if gen > 1:
	# 	for i in range(args.pop_size):
	# 		valid_start = time.time()
	# 		population.get_population()[i+args.pop_size].objs.reset()
	# 		population.get_population()[i+args.pop_size].top1.reset()
	# 		population.get_population()[i+args.pop_size].top5.reset()
	#
	# if gen == 1:
	for i in range(len(population.get_population())):
		valid_start = time.time()
		population.get_population()[i].objs.reset()
		population.get_population()[i].top1.reset()
		population.get_population()[i].top5.reset()

	with torch.no_grad():
		for step, (inputs, targets) in enumerate(valid_queue):
			discrete_alphas = utils.discretize(population.get_population()[step % population.get_population_size()].arch_parameters, device)
			model.copy_arch_parameters(discrete_alphas)
			n = inputs.size(0)
			inputs = inputs.to(device)
			targets = targets.to(device)
			logits = model(inputs)
			loss = criterion(logits, targets)
		
			prec1, prec5 = utils.accuracy(logits, targets, topk = (1, 5))
			population.get_population()[step % population.get_population_size()].objs.update(loss.data, n)
			population.get_population()[step % population.get_population_size()].top1.update(prec1.data, n)
			population.get_population()[step % population.get_population_size()].top5.update(prec5.data, n)

	if gen == 1:
		population.pop_sort()
		for i in range(args.pop_size):
			discrete_alphas = utils.discretize(population.get_population()[i].arch_parameters, device)
			model.copy_arch_parameters(discrete_alphas)
			genotype = model.genotype()
			logging.info('genotype = %s', genotype)
			logging.info("[{} Generation] {}/{} finished with validation loss: {}, prec1: {}, prec5: {}, scaling factor: {}, encoding: {}".format(gen, i+1, len(population.get_population()),
																												population.get_population()[i].objs.avg,
																												population.get_population()[i].top1.avg,
																												population.get_population()[i].top5.avg,
																												population.get_population()[i].get_mutate_factor(),
																												discrete_alphas))
	if gen > 1:
		pop_index = []
		for i in range(args.pop_size):
			# print('population-{}'.format(population.get_population()[i].arch_parameters))
			if population.get_population()[i].top1.avg > population.get_population()[i+args.pop_size].top1.avg:
				pop_index.append(i+args.pop_size)
			else:
				pop_index.append(i)
		pop_index.sort()
		population.pop_pop(pop_index)
		population.pop_sort()
		for i in range(args.pop_size):
			# print('population-{}'.format(population.get_population()[i].arch_parameters))
			discrete_alphas = utils.discretize(population.get_population()[i].arch_parameters, device)
			# print('population-{}'.format(population.get_population()[i].arch_parameters))
			# print('discrete_alphas = %s', discrete_alphas)
			model.copy_arch_parameters(discrete_alphas)
			genotype = model.genotype()
			logging.info('genotype = %s', genotype)
			logging.info(
				"[{} Generation] {}/{} finished with validation loss: {}, prec1: {}, prec5: {}, scaling factor: {}, encoding: {}".format(gen, i + 1, len(population.get_population()),
																									    population.get_population()[i].objs.avg,
																									    population.get_population()[i].top1.avg,
																									    population.get_population()[i].top5.avg,
																										population.get_population()[i].get_mutate_factor(),
																										discrete_alphas))

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device("cuda:{}".format(args.gpu))
cpu_device = torch.device("cpu")

torch.cuda.set_device(args.gpu)
cudnn.deterministic = True
cudnn.enabled = True
cudnn.benchmark = False

CIFAR_CLASSES = 10
# genotype = eval("genotypes.%s" % "DARTS")
criterion = nn.CrossEntropyLoss()
criterion.to(device)
model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, device)
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum = args.momentum, weight_decay = args.weight_decay)

train_transform, valid_transform = utils._data_transforms_cifar10(args)
train_data = dset.CIFAR10(root = args.data, train = True, download = True, transform = train_transform)

num_train = len(train_data)
indices = list(range(num_train))
split = int(np.floor(args.train_portion * num_train))

train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, pin_memory = False, num_workers = 2,
														sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split]))
valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size = args.batch_size,
      sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory = False, num_workers = 2)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min = args.learning_rate_min)

k = sum(1 for i in range(model._steps) for _ in range(2 + i))
num_ops = len(genotypes.PRIMITIVES)

## Creating Population
population = Population(args.pop_size, model._steps, device)

DIR = "search-{}".format(time.strftime("%Y%m%d-%H%M%S"))
if args.dir is not None:
	if not os.path.exists(args.dir):
		utils.create_exp_dir(args.dir)
	DIR = os.path.join(args.dir, DIR)
else:
	DIR = os.path.join(os.getcwd(), DIR)
utils.create_exp_dir(DIR)
utils.create_exp_dir(os.path.join(DIR, "weights"))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(DIR, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info("gpu device = {}".format(args.gpu))
logging.info("args =  %s", args)
logging.info("[INFO] Using ga with dicretization")
logging.info("num_train: {}, split: {}".format(num_train, split))

ga = GeneticAlgorithm(args.pop_size, args.tsize, device, args.mutate_rate)
writer = SummaryWriter(os.path.join(DIR, 'runs', 'ga'))

#scheduler.step()
lr = scheduler.get_lr()[0]

# STAGE 1
start = time.time()
for epoch in range(args.warm_up):
	start_time = time.time()
	logging.info('[INFO] Warming up!!!')
	warm_train(model, train_queue, criterion, optimizer)
	logging.info("[INFO] Warming up finished in {} minutes".format((time.time() - start_time) / 60))


for epoch in range(args.epochs):
	## Training the whole population
	logging.info("[INFO] Generation {} training with learning rate {}".format(epoch + 1, scheduler.get_lr()[0]))
	start_time = time.time()

	train(model, train_queue, criterion, optimizer, epoch + 1)
	logging.info("[INFO] Training finished in {} minutes".format((time.time() - start_time) / 60))
	torch.save(model.state_dict(), "model.pt")
	#lr = scheduler.get_lr()[0]
	scheduler.step()

	logging.info("[INFO] Evaluating Generation {} ".format(epoch + 1))
	validation(model, valid_queue, criterion, epoch + 1, population)
	# population.pop_sort()
	# population.get_population_top50()

	for i, p in enumerate(population.get_population()):
		writer.add_scalar("pop_top1_{}".format(i + 1), p.get_fitness(), epoch + 1)
		writer.add_scalar("pop_top5_{}".format(i + 1), p.top5.avg, epoch + 1)
		writer.add_scalar("pop_obj_valid_{}".format(i + 1), p.objs.avg, epoch + 1)

	with open(os.path.join(DIR, "population_{}.pickle".format(epoch + 1)), 'wb') as f:
		pickle.dump(population, f)

	# if epoch == args.epochs - 1:
	# 	# normal_cell = utils.derive_architecture(population.get_population()[0].arch_parameters[0])
	# 	# reduction_cell = utils.derive_architecture(population.get_population()[0].arch_parameters[1])
	# 	# normal_cell = utils.discretize()
	# 	# concat = [2, 3, 4, 5]
	# 	# genotype = genotypes.Genotype(normal = normal_cell, normal_concat = concat, reduce =  reduction_cell, reduce_concat = concat)
	# 	model.copy_arch_parameters(population.get_population()[0].arch_parameters)
	# 	assert utils.check_equality(model, population.get_population()[0].arch_parameters)
	# 	genotype2 = model.genotype()
	# 	# assert genotype2.normal == genotype.normal
	# 	# assert genotype2.reduce == genotype.reduce
	# 	with open(os.path.join(DIR, "genotype.pickle"), 'wb') as f:
	# 		pickle.dump(genotype, f)
	# 	logging.info("[INFO] Saving the best architecture after {} generation".format(epoch + 1))
	# 	logging.info("The current best individual: {}".format(genotype))
	# 	population.print_population()
	# Applying Genetic Algorithm
	pop = ga.evolve(population, epoch, args.epochs)
	# pop.get_population_top50()
	population = pop

	last = time.time() - start_time
	logging.info("[INFO] {}/{} epoch finished in {} minutes".format(epoch + 1, args.epochs, last / 60))
	utils.save(model, os.path.join(DIR, "weights","weights.pt"))

writer.close()

last = time.time() - start
logging.info("[INFO] {} hours".format(last / 3600))


