import argparse
import multiprocessing
import os

import torch

DATASETS = ['femnist', 'bosch', 'cifar10', 'emnist', 'mnist']
ATTACKS = ['none', 'data', 'free_rider', 'model', 'colluding']
METHODS = [f[:1].upper() + f[1:-3] for f in os.listdir('trainers')]


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', help='dataset;', type=str, choices=DATASETS, default='femnist')
    parser.add_argument('--method', help='name of method;', choices=METHODS, type=str, default='Autogmfl')
    parser.add_argument('--attack', help='Attacking option', choices=ATTACKS, type=str, default='none')
    parser.add_argument('--model', help='name of model;', type=str, default='cnn')
    parser.add_argument('--global_rounds', help='number of rounds to simulate;', type=int, default=0)
    parser.add_argument('--eval_every', help='evaluate every ____ rounds;', type=int, default=50)
    parser.add_argument('--clients_per_round', help='number of clients trained per round;', type=int, default=32)
    parser.add_argument('--batch_size', help='batch size when clients train on data;', type=int, default=32)
    parser.add_argument('--seed', help='seed for random client sampling and batch splitting', type=int, default=0)
    parser.add_argument('--metrics_name', help='name for metrics file;', type=str, default='none', required=False)
    parser.add_argument('--metrics_dir', help='dir for metrics file;', type=str, default='metrics', required=False)
    parser.add_argument('-pc', help='proportion of clients to be corrupted', type=float, default=0.5)
    parser.add_argument('-ps', help='proportion of sample in each client to be corrupted', type=float, default=1.)
    
    parser.add_argument('--num_actors', help='total number of actors', type=int, default=0)
    parser.add_argument('--num_gpus', help='number of gpus', type=int, default=2)
    parser.add_argument('--lamb1', help='lamb1', type=float, default=1.0)
    parser.add_argument('--lamb2', help='lamb2', type=float, default=1.0)
    epoch_capability_group = parser.add_mutually_exclusive_group()
    epoch_capability_group.add_argument('--local_rounds', help='number of local epochs', type=int, default=50)
    parser.add_argument('--lr', help='learning rate for local optimizers;', type=float, default=0.05, required=False)
    
    flag_parser = parser.add_mutually_exclusive_group(required=False)
    flag_parser.add_argument('--decay', dest='decay', action='store_true')
    flag_parser.add_argument('--no-decay', dest='decay', action='store_false')
    
    load_flag_parser = parser.add_mutually_exclusive_group(required=False)
    load_flag_parser.add_argument('--loadmodel', dest='loadmodel', action='store_true')
    load_flag_parser.add_argument('--no-loadmodel', dest='loadmodel', action='store_false')
    
    scale_flag_parser = parser.add_mutually_exclusive_group(required=False)
    scale_flag_parser.add_argument('--scale', dest='scale', action='store_true')
    scale_flag_parser.add_argument('--no-scale', dest='scale', action='store_false')
    
    parser.set_defaults(decay=False)
    parser.set_defaults(loadmodel=False)
    parser.set_defaults(scale=False)
    return parser.parse_args()


options = parse_args()

MODEL_PARAMS = {
    'femnist.cnn': (62, options.lr, 128),  # num_classes, test_batch_size
    'bosch.mlp': (100, options.lr, 128),  # seq_len, num_classes, num_hidden, test_batch_size
}

if options.global_rounds == 0:
    options.global_rounds = 20000 // options.local_rounds

if options.num_actors == 0:
    options.num_actors = min(multiprocessing.cpu_count(), 20)
options.model_params = MODEL_PARAMS['%s.%s' % (options.dataset, options.model)]
options.model_path = '%s.%s.%s' % ('models', options.dataset, options.model)
options.optimizer_path = '%s.%s' % ('trainers', options.method.lower())

if not torch.cuda.is_available():
    options.gpu_per_actor = 0
else:
    options.gpu_per_actor = (options.num_gpus - 0.05) / options.num_actors

if not os.path.exists(options.metrics_dir):
    os.mkdir(options.metrics_dir)
if options.metrics_name == 'none':
    options.metrics_name = '_'.join([options.dataset, options.method, options.attack, str(int(options.pc * 100)),
                                     str(options.lamb2), str(options.lamb1), str(options.seed)])

LOG_FILENAME = os.path.join(options.metrics_dir, options.metrics_name + "_state.csv")

options.log_filename = os.path.join(options.metrics_dir, options.metrics_name + '.txt')

print(options)
