import os
import sys
import logging
import random
import torch.nn as nn
import genotypes
import argparse
import numpy as np
import pickle
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torch.utils
import torch.nn.functional as F
import time
import utils
import pandas as pd

from cell_operationsNAS201   import NAS_BENCH_201
from config_utils            import load_config
from datasets                import get_datasets, get_nas_search_loaders
from deNAS201                import DifferentialEvolution
from nas_201_api             import NASBench201API as API

from populationNAS201        import *
from optimizers              import get_optim_scheduler
from search_model_NAS201     import TinyNetwork
from torch.utils.tensorboard import SummaryWriter
from torch.autograd          import Variable

parser = argparse.ArgumentParser("NAS201")
parser.add_argument('--data', type = str, default = '/fasterdatasets/cifar-10', help = 'location of the data corpus')
parser.add_argument('--dir', type = str, default = None, help = 'location of trials')
parser.add_argument('--cutout', action = 'store_true', default = False, help = 'use cutout')
parser.add_argument('--cutout_length', type = int, default = 16, help = 'cutout length')
parser.add_argument('--batch_size', type = int, default = 64, help = 'batch size')
parser.add_argument('--valid_batch_size', type = int, default = 1024, help = 'validation batch size')
parser.add_argument('--epochs', type = int, default = 50, help = 'num of training epochs')
parser.add_argument('--seed', type = int, default = 1, help = 'random seed')
parser.add_argument('--gpu', type = int, default = 5, help = 'gpu device id')
parser.add_argument('--tsize', type = int, default = 10, help = 'Tournament size')
parser.add_argument('--num_elites', type = int, default = 1, help = 'Number of Elites')
parser.add_argument('--mutate_rate', type = float, default = 0.1, help = 'mutation rate')
parser.add_argument('--learning_rate', type = float, default = 0.025, help = 'init learning rate')
parser.add_argument('--learning_rate_min', type = float, default = 0.001, help = 'min learning rate')
parser.add_argument('--momentum', type = float, default = 0.9, help = 'momentum')
parser.add_argument('--weight_decay', type = float, default = 3e-4, help = 'weight decay')
parser.add_argument('--grad_clip', type = float, default = 5, help = 'gradient clipping')
parser.add_argument('--pop_size', type = int, default = 50, help = 'population size')
parser.add_argument('--report_freq', type = float, default = 50, help = 'report frequency')
parser.add_argument('--init_channels', type = int, default = 16, help = 'num of init channels')

# Added for NAS201
#parser.add_argument('--channel', type = int, default = 16, help = 'initial channel for NAS201 network')
parser.add_argument('--num_cells', type = int, default = 5, help = 'number of cells for NAS201 network')
parser.add_argument('--max_nodes', type = int, default = 4, help = 'maximim nodes in the cell for NAS201 network')
parser.add_argument('--track_running_stats', action = 'store_true', default = False, help = 'use track_running_stats in BN layer')
parser.add_argument('--dataset', type = str, default = 'cifar10', help = '["cifar10", "cifar100", "ImageNet16-120"]')
parser.add_argument('--api_path', type = str, default = '../NAS-Bench-102-v1_0-e61699.pth', help = '["cifar10", "cifar10-valid","cifar100", "imagenet16-120"]')
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
parser.add_argument('--config_path', type=str, default='./configs/CIFAR.config', help='The config path.')
args = parser.parse_args()

def get_arch_score(api, arch_index, dataset, hp, acc_type):
  info = api.query_by_index(arch_index, hp = str(hp))
  return info.get_metrics(dataset, acc_type)['accuracy']

def train(model, train_queue, criterion, optimizer, gen):
  model.train()
  for step, (inputs, targets) in enumerate(train_queue):
    #model.copy_arch_parameters(population.get_population()[step % args.pop_size].arch_parameters)
    #assert utils.check_equality(model, population.get_population()[step % args.pop_size].arch_parameters)
    #discrete_alphas = utils.discretize(population.get_population()[step % args.pop_size].arch_parameters, device)
   
    #Copying and checking the discretized alphas
    model.update_alphas(population.get_population()[step % args.pop_size].arch_parameters[0])
    discrete_alphas = model.discretize()
    _, df_max, _ = model.show_alphas_dataframe()
    assert np.all(np.equal(df_max.to_numpy(), discrete_alphas.cpu().numpy()))
    assert model.check_alphas(discrete_alphas)
    
    n = inputs.size(0)
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    #inputs = inputs.cuda(non_blocking=True)
    #targets = targets.cuda(non_blocking=True)
    optimizer.zero_grad()
    _, logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, targets, topk = (1, 5))
    population.get_population()[step % args.pop_size].objs.update(loss.data.cpu().item(), n)
    population.get_population()[step % args.pop_size].top1.update(prec1.data.cpu().item(), n)
    population.get_population()[step % args.pop_size].top5.update(prec5.data.cpu().item(), n)
    
    #population.get_population()[step % args.pop_size].accumulate()
  
    #print(step)
    if (step + 1) % 100 == 0:
    #  break
      logging.info("[{} Generation]".format(gen))
      logging.info("Using Training batch #{} for {}/{} architecture with loss: {}, prec1: {}, prec5: {}".format(step, step % args.pop_size, 
                                              len(population.get_population()), 
                                              population.get_population()[step % args.pop_size].objs.avg, 
                                              population.get_population()[step % args.pop_size].top1.avg, 
                                              population.get_population()[step % args.pop_size].top5.avg))
    #break

def validation(model, valid_queue, criterion, gen):
  model.eval()
  for i in range(len(population.get_population())):
    valid_start = time.time()
    #discrete_alphas = utils.discretize(population.get_population()[i].arch_parameters, device)
    #model.copy_arch_parameters(discrete_alphas)
    #assert utils.check_equality(model, discrete_alphas)
    
    #Copying and checking the discretized alphas
    model.update_alphas(population.get_population()[i].arch_parameters[0])
    discrete_alphas = model.discretize()
    _, df_max, _ = model.show_alphas_dataframe()
    assert np.all(np.equal(df_max.to_numpy(), discrete_alphas.cpu().numpy()))
    assert model.check_alphas(discrete_alphas)
   
    population.get_population()[i].objs.reset()
    population.get_population()[i].top1.reset()
    population.get_population()[i].top5.reset()
    with torch.no_grad():
      for step, (inputs, targets) in enumerate(valid_queue):
        n = inputs.size(0)
        inputs = inputs.to(device)
        targets = targets.to(device)
        _, logits = model(inputs)
        loss = criterion(logits, targets)
    
        prec1, prec5 = utils.accuracy(logits, targets, topk = (1, 5))
        population.get_population()[i].objs.update(loss.data.cpu().item(), n)
        population.get_population()[i].top1.update(prec1.data.cpu().item(), n)
        population.get_population()[i].top5.update(prec5.data.cpu().item(), n)
      
        #print(step)
        #if (step + 1) % 10 == 0:
        #break
    #print("Finished in {} seconds".format((time.time() - valid_start) ))
    #break
  if gen > 1:
    pop_index = []
    for i in range(args.pop_size):
      # print('population-{}'.format(population.get_population()[i].arch_parameters))
      if population.get_population()[i].top1.avg > population.get_population()[i + args.pop_size].top1.avg:
        pop_index.append(i + args.pop_size)
      else:
        pop_index.append(i)
    pop_index.sort()
    population.pop_pop(pop_index)

  for i in range(len(population.get_population())):
    logging.info("[{} Generation] {}/{} finished with validation loss: {}, prec1: {}, prec5: {}".format(gen, i + 1, len(
      population.get_population()),
                                                                                                        population.get_population()[
                                                                                                          i].objs.avg,
                                                                                                        population.get_population()[
                                                                                                          i].top1.avg,
                                                                                                        population.get_population()[
                                                                                                          i].top5.avg))

DIR = "search-{}-{}".format(time.strftime("%Y%m%d-%H%M%S"), args.dataset)
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

# Initializing the summary writer
writer = SummaryWriter(os.path.join(DIR, 'runs'))

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

assert args.api_path is not None, 'NAS201 data path has not been provided'
api = API(args.api_path, verbose = False)
logging.info(f'length of api: {len(api)}')

# Configuring dataset and dataloader
if args.dataset == 'cifar10':
  acc_type     = 'ori-test'
  val_acc_type = 'x-valid'
else:
  acc_type     = 'x-test'
  val_acc_type = 'x-valid'

datasets = ['cifar10', 'cifar100', 'ImageNet16-120']
assert args.dataset in datasets, 'Incorrect dataset'
if args.cutout:
  train_data, valid_data, xshape, num_classes = get_datasets(name = args.dataset, root = args.data, cutout=args.cutout)
else:
  train_data, valid_data, xshape, num_classes = get_datasets(name = args.dataset, root = args.data, cutout=-1)
logging.info("train data len: {}, valid data len: {}, xshape: {}, #classes: {}".format(len(train_data), len(valid_data), xshape, num_classes))

config = load_config(path=args.config_path, extra={'class_num': num_classes, 'xshape': xshape}, logger=None)
logging.info(f'config: {config}')
_, train_loader, valid_loader = get_nas_search_loaders(train_data=train_data, valid_data=valid_data, dataset=args.dataset,
                                                        config_root='configs', batch_size=(args.batch_size, args.valid_batch_size),
                                                        workers=args.workers)
train_queue, valid_queue = train_loader, valid_loader
logging.info('search_loader: {}, valid_loader: {}'.format(len(train_queue), len(valid_queue)))

# Model Initialization
#model_config = {'C': 16, 'N': 5, 'num_classes': num_classes, 'max_nodes': 4, 'search_space': NAS_BENCH_201, 'affine': False}
model = TinyNetwork(C = args.init_channels, N = args.num_cells, max_nodes = args.max_nodes,
                    num_classes = num_classes, search_space = NAS_BENCH_201, affine = False,
                    track_running_stats = args.track_running_stats)
model = model.to(device)
#logging.info(model)

optimizer, _, criterion = get_optim_scheduler(parameters=model.get_weights(), config=config)
criterion = criterion.cuda()
logging.info(f'optimizer: {optimizer}\nCriterion: {criterion}')

# logging the initialized architecture
best_arch_per_epoch = []

arch_str = model.genotype().tostr()
arch_index = api.query_index_by_arch(model.genotype())
if args.dataset == 'cifar10':
  test_acc = get_arch_score(api, arch_index, 'cifar10', 200, acc_type)
  valid_acc = get_arch_score(api, arch_index, 'cifar10-valid', 200, val_acc_type)
  writer.add_scalar("test_acc", test_acc, 0)
  writer.add_scalar("valid_acc", valid_acc, 0)
else:
  test_acc = get_arch_score(api, arch_index, args.dataset, 200, acc_type)
  valid_acc = get_arch_score(api, arch_index, args.dataset, 200, val_acc_type)
  writer.add_scalar("test_acc", test_acc, 0)
  writer.add_scalar("valid_acc", valid_acc, 0)
tmp = (arch_str, test_acc, valid_acc)
best_arch_per_epoch.append(tmp)

'''
optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum = args.momentum, weight_decay = args.weight_decay)
criterion = nn.CrossEntropyLoss()
criterion.to(device)
train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, pin_memory = False, num_workers = 2,
                            sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split]))
valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size = 1024, #args.batch_size,
      sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory = False, num_workers = 2)
'''
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min = args.learning_rate_min)
logging.info(f'Scheduler: {scheduler}')

## Creating Population
population = Population(pop_size = args.pop_size, num_edges = model.get_alphas()[0].shape[0], device = device)

logging.info(f'torch version: {torch.__version__}, torchvision version: {torch.__version__}')
logging.info("gpu device = {}".format(args.gpu))
logging.info("args =  %s", args)
logging.info("[INFO] Using de with dicretization")

de = DifferentialEvolution(args.pop_size, args.tsize, device, args.mutate_rate)

#scheduler.step()
lr = scheduler.get_lr()[0]

# STAGE 1
start = time.time()
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
  validation(model, valid_queue, criterion, epoch + 1)

  # Sorting the population according to the fitness in decreasing order
  population.pop_sort()
  
  for i, p in enumerate(population.get_population()):
    writer.add_scalar("pop_top1_{}".format(i + 1), p.get_fitness(), epoch + 1)
    writer.add_scalar("pop_top5_{}".format(i + 1), p.top5.avg, epoch + 1)
    writer.add_scalar("pop_obj_valid_{}".format(i + 1), p.objs.avg, epoch + 1)

  # Saving the population after each generation
  tmp = []
  for individual in population.get_population():
    tmp.append(tuple((individual.arch_parameters[0].cpu().numpy(), individual.get_fitness())))
  with open(os.path.join(DIR, "population_{}.pickle".format(epoch + 1)), 'wb') as f:
    pickle.dump(tmp, f)

  # Copying the best individual to the model
  model.update_alphas(population.get_population()[0].arch_parameters[0])
  assert model.check_alphas(population.get_population()[0].arch_parameters[0])
  arch_str = model.genotype().tostr()
  arch_index = api.query_index_by_arch(model.genotype())
  if args.dataset == 'cifar10':
    test_acc = get_arch_score(api, arch_index, 'cifar10', 200, acc_type)
    valid_acc = get_arch_score(api, arch_index, 'cifar10-valid', 200, val_acc_type)
    writer.add_scalar("test_acc", test_acc, epoch + 1)
    writer.add_scalar("valid_acc", valid_acc, epoch + 1)
  else:
    test_acc = get_arch_score(api, arch_index, args.dataset, 200, acc_type)
    valid_acc = get_arch_score(api, arch_index, args.dataset, 200, val_acc_type)
    writer.add_scalar("test_acc", test_acc, epoch + 1)
    writer.add_scalar("valid_acc", valid_acc, epoch + 1)
  tmp = (arch_str, test_acc, valid_acc)
  best_arch_per_epoch.append(tmp)
  
  # Applying Differential Evolution
  pop = de.evolve(population)
  population = pop 
  
  last = time.time() - start_time
  logging.info("[INFO] {}/{} epoch finished in {} minutes".format(epoch + 1, args.epochs, last / 60))
  utils.save(model, os.path.join(DIR, "weights","weights.pt"))
  
  #if epoch > 0:
  #  break

writer.close()

last = time.time() - start
logging.info("[INFO] {} hours".format(last / 3600))

for i in range(len(population.get_population())):
  logging.info(f'[INFO] {i}-th Best Architecture after the search: {best_arch_per_epoch[i]}')
  # logging.info(f'length best_arch_per_epoch: {len(best_arch_per_epoch)}')
  # with open(os.path.join(DIR, "best_architectures.pickle"), 'wb') as f:
  #   pickle.dump(best_arch_per_epoch, f)

