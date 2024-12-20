import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import genotypes
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel as DDP
from model import NetworkImageNet as Network

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default='/fasterdatasets/', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='imagenet', help='imagenet')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--workers', type=int, default=20, help='number of workers to load dataset')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--arch', type=str, default='test', help='which architecture to use')
parser.add_argument('--reload_pth', type=str, default=None, help='reload pth')
parser.add_argument('--lr_scheduler', type=str, default='cosine', help='lr scheduler')
parser.add_argument('--world_size', type=int, default=2, help='world size')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CLASSES = 1000


class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss

def main(rank):
  dist.init_process_group('nccl', rank=rank, world_size=args.world_size)
  torch.cuda.set_device(rank)

  # fix seeds
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  os.environ['PYTHONHASHSEED']=str(args.seed)
  cudnn.enabled = True
  cudnn.benchmark = True
  torch.cuda.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)
  torch.backends.cudnn.deterministic = True
  num_gpus = torch.cuda.device_count()
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  logging.info('genotype = %s', genotype)
  model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
  model = model.cuda()
  model = DDP(model, device_ids=[rank])
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))


  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
  criterion_smooth = criterion_smooth.cuda()

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  if args.reload_pth is not None:
    checkpoint = torch.load(args.reload_pth)
    start_epoch = checkpoint['epoch']
    best_valid_acc = checkpoint['best_valid_acc']
    best_valid_acc_r5 = checkpoint['best_valid_acc_r5']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logging.info("=> loaded checkpoint '{}' (epoch {})".format(args.reload_pth, checkpoint['epoch']))
  else:
    start_epoch = 0
    best_acc_top1 = 0
    best_acc_top5 = 0
    logging.info("=> no checkpoint found at '{}'".format(args.reload_pth))

  data_dir = os.path.join(args.data, 'imagenet2012')
  traindir = os.path.join(data_dir, 'train')
  validdir = os.path.join(data_dir, 'val')
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  train_data = dset.ImageFolder(
    traindir,
    transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.2),
      transforms.ToTensor(),
      normalize,
    ]))
  valid_data = dset.ImageFolder(
    validdir,
    transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
    ]))

  train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
  valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data, shuffle=True)

  train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, sampler=train_sampler, pin_memory=True, num_workers=args.workers)

  valid_queue = torch.utils.data.DataLoader(
    valid_data, batch_size=args.batch_size, sampler=valid_sampler, pin_memory=True, num_workers=args.workers)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

  best_valid_acc = 0.0
  for epoch in range(start_epoch, args.epochs):
    if args.lr_scheduler == 'cosine':
      scheduler.step(epoch)
      current_lr = scheduler.get_lr()[0]
    elif args.lr_scheduler == 'linear':
      current_lr = adjust_lr(optimizer, epoch)
    else:
      print('Wrong lr type, exit')
      sys.exit(1)
    logging.info('epoch: %d lr %e', epoch, current_lr)
    if epoch < 5 and args.batch_size > 256:
      for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr * (epoch + 1) / 5.0
      logging.info('warming up epoch: %d lr %e', epoch, current_lr * (epoch + 1) / 5.0)
    if num_gpus > 1:
      model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
    else:
      model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
    epoch_start = time.time()
    train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer)
    logging.info('train_acc %f', train_acc)

    with torch.no_grad():
      valid_acc, valid_acc_r5, valid_obj = infer(valid_queue, model, criterion)
      epoch_duration = time.time() - epoch_start
      logging.info('valid_acc %f, valid_acc_r5 %f', valid_acc, valid_acc_r5)

      if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        best_valid_acc_r5 = valid_acc_r5
        utils.save(model, os.path.join(args.save, 'best_weights.pt'))
        torch.save({
          'epoch': epoch + 1,
          'state_dict': model.state_dict(),
            'best_valid_acc': best_valid_acc,
            'best_valid_acc_r5': best_valid_acc_r5,
        }, os.path.join(args.save, 'best_model.pth'))
      logging.info('best_valid_acc %f, best_valid_acc_r5 %f, epoch time: %ds.', best_valid_acc, best_valid_acc_r5, epoch_duration)

    # utils.save(model, os.path.join(args.save, 'weights.pt'))

def adjust_lr(optimizer, epoch):
    # Smaller slope for the last 5 epochs because lr * 1/250 is relatively large
    if args.epochs - epoch > 5:
      lr = args.learning_rate * (args.epochs - 5 - epoch) / (args.epochs - 5)
    else:
      lr = args.learning_rate * (args.epochs - epoch) / ((args.epochs - 5) * 5)
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr
    return lr


def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  batch_time = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    #input = Variable(input).cuda()
    #target = Variable(target).cuda(async=True)
    input = input.cuda()
    target = target.cuda(non_blocking=True)

    b_start = time.time()

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    batch_time.update(time.time()-b_start)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      end_time = time.time()
      if step == 0:
        duration = 0
        start_time = time.time()
      else:
        duration = end_time - start_time
        start_time = time.time()
      logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f Duration: %ds BTime: %.3fs',
                   step, objs.avg, top1.avg, top5.avg, duration, batch_time.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = input.cuda()
    target = target.cuda(non_blocking=True)

    logits, _ = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      end_time = time.time()
      if step == 0:
        duration = 0
        start_time = time.time()
      else:
        duration = end_time - start_time
        start_time = time.time()
      logging.info('VALID Step: %03d Objs: %e R1: %f R5: %f Duration: %ds', step, objs.avg, top1.avg, top5.avg,
                   duration)

  return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
  size = args.world_size
  processes = []
  for rank in range(size):
    p = Process(target=main, args=(rank,))
    p.start()
    processes.append(p)

  for p in processes:
    p.join()