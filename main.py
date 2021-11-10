import importlib
import os
import random
import time

import numpy as np
import ray
import torch

from utils.args import options
from utils.data_utils import read_data


def set_random_seed(seed_value=0):
    np.random.seed(seed_value)  # cpu vars
    random.seed(seed_value)  # Python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # Python hash buildin
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = options
    if not ray.is_initialized():
        ray.init(include_dashboard=False, num_gpus=args.num_gpus)
    set_random_seed(options.seed)
    
    if os.path.exists(options.log_filename):
        os.remove(options.log_filename)
    
    train_data_dir = os.path.join('data', args.dataset, 'train')
    test_data_dir = os.path.join('data', args.dataset, 'test')
    t_start = time.time()
    dataset = read_data(train_data_dir, test_data_dir)
    # print(args.model_path)
    mod = importlib.import_module(args.model_path)
    opt = importlib.import_module(args.optimizer_path)
    net = getattr(mod, 'Net')
    Client = getattr(opt, args.method + 'Client')
    Model = getattr(opt, args.method + 'Model')
    server = getattr(opt, args.method + 'Server')(options, net, dataset, Client, Model)
    
    print('Loading time cost: ', time.time() - t_start)
    server.run()


if __name__ == '__main__':
    main()
