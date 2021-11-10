import copy
import logging
import os
import time
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from models.actormanager import ActorManager
from utils.data_utils import read_data, preprocess_data


class BaseClient:
    
    def __init__(self, client_id, train_data, test_data, seed=0, attack=False, attack_type=0):
        self.id = client_id
        self.is_corrupted = False
        self.attack_type = -1
        self.loss = {}
        self.raw_train_data = train_data
        self.raw_test_data = test_data
        self.seed = seed
        self.noise = 0
        self.scale = 1.0
        self.free_rider = False
        self.colluding = False
        
    def train(self, model, epochs, g_net, scale=False, **kwargs):
        data = [next(self.train_data) for _ in range(epochs)]
        
        if scale:
            return model.update.remote(data, epochs, g_net, self.local_model,
                                       noise=self.noise, scale=self.scale, free_rider = self.free_rider, **kwargs)
        else:
            return model.update.remote(data, epochs, g_net, self.local_model,
                                       noise=self.noise, scale=1.0,  free_rider = self.free_rider, **kwargs)
    
    # def train(self, model, epochs, g_net, **kwargs):
    #     data = [next(self.train_data) for _ in range(epochs)]
    #     return model.update.remote(data, epochs, g_net, self.local_model, **kwargs)
    
    def set_datasets(self, batch_size=32):
        # self.train_data = self._load_data(self.raw_train_data)
        self.train_data = preprocess_data(np.array(self.raw_train_data['x']), np.array(self.raw_train_data['y']),
                                          batch_size=batch_size)
        self.test_data = self._load_data(self.raw_test_data)
    
    def _load_data(self, data, batch_size=32):
        tensor_x = torch.Tensor(data['x'])  # transform to torch tensor
        tensor_y = torch.LongTensor(data['y'])
        
        dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
        dataloader = DataLoader(dataset, batch_size=batch_size)  # create your dataloader
        return dataloader
    
    def poison_data(self, p, L):
        """
            poison_by_shuffling_labels selects a fraction p of the samples and
            shuffles their labels randomly
        """
        sz = len(self.raw_train_data['y'])
        self.raw_train_data['y'] = np.array(self.raw_train_data['y'])
        n_poisoned = int(sz * p)
        poisoned_points = np.random.choice(sz, n_poisoned, replace=False)
        reordered = np.random.permutation(poisoned_points)
        self.raw_train_data['y'][poisoned_points] = self.raw_train_data['y'][reordered]
        # self.raw_train_data['y'][poisoned_points] = L - self.raw_train_data['y'][poisoned_points] - 1
        
        return self.raw_train_data
    
    def poison_model(self, max_noise=0.1, colluding=False):
        self.attack_type = 'model'
        if colluding:
            self.noise = 0.05
        else:
            self.noise = np.random.uniform(0.0, max_noise)
    
    def poison_by_noise(self, args):
        
        x = np.array(self.train_data['x'])
        x_new = x
        if args.dataset == 'femnist':
            scale = 0.7
            noise = np.random.normal(0, scale, x.shape)
            x_noisy = x + noise
            x_new = (x_noisy - np.min(x_noisy)) / (np.max(x_noisy) - np.min(x_noisy))
            
            # img = np.array(x_new[0]).reshape((28, 28))
            # plt.imshow(img, cmap='gray', aspect='equal')
            # plt.grid(False)
            # _ = plt.show()
        # modify client in-place
        self.train_data['x'] = x_new
    
    def test(self, model):
        return model.test_model.remote(self.test_data)
    
    @property
    def num_test_samples(self):
        if self.test_data is None:
            return 0
        return len(self.test_data.dataset)
    
    @property
    def num_train_samples(self):
        if self.train_data is None:
            return 0
        return len(self.raw_train_data['y'])
    
    @property
    def num_samples(self):
        return self.num_train_samples + self.num_test_samples


class BaseServer(object):
    __metaclass__ = ABCMeta
    
    def __init__(self, params, net, dataset, client, model=None, num_class=10):
        self.dataset = dataset
        for key, val in vars(params).items(): setattr(self, key, val)
        self.clients = self.setup_clients(client)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('{} Clients in Total'.format(len(self.clients)))
        self.model = model(net, 0, num_class, 0.01, 64)
        self.train_proxy = ActorManager(params, model)
        self.global_model = self.train_proxy.init_actors(net)
        self.init_global_model = copy.deepcopy(self.global_model)
        self.corrupted_counter = 0
        self.corrupt_clients()
        self.time_begin = time.time()
        logging.basicConfig(filename=self.log_filename, level=logging.INFO, format='%(message)s')
    
    def select_clients(self, num_clients=20, weighted=False):
        possible_clients = np.array(self.clients)
        num_clients = min(num_clients, len(possible_clients))
        candidates = [i for i in range(len(possible_clients))]
        if weighted:
            p = np.array([client.num_train_samples for client in self.clients]) / \
                np.sum([client.num_train_samples for client in self.clients])
            selected_index = np.random.choice(candidates, num_clients, p=p, replace=False)
        else:
            selected_index = np.random.choice(candidates, num_clients, replace=False)
        self.selected_indexes = selected_index
        
        self.selected_clients = possible_clients[selected_index]
        return self.selected_clients
    
    def corrupt_clients(self):
        # Randomly attack clients
        pc = self.pc
        ps = self.ps
        att_type = self.attack
        n = int(len(self.clients) * pc)
        np.random.seed(self.seed)
        selected_indexes = np.random.choice(range(len(self.clients)), n, replace=False)
        selected_indexes = np.sort(selected_indexes)
        for i in selected_indexes:
            self.clients[i].is_corrupted = True
            self.clients[i].attack_type = att_type
            if att_type == 'data':
                self.clients[i].poison_data(ps, 62)
            elif att_type == 'model':
                self.clients[i].poison_model(max_noise=0.1)
            elif att_type == 'free_rider':
                self.clients[i].free_rider = True
            elif att_type == 'colluding':
                self.clients[i].colluding = True
        print("attacked clients: " + ','.join([str(i) for i in selected_indexes]))
    
    def setup_clients(self, Client):
        
        train_data_dir = os.path.abspath(os.path.join('data', self.dataset, 'train'))
        test_data_dir = os.path.abspath(os.path.join('data', self.dataset, 'test'))
        users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
        print(users)
        self.clients = []
        for i, u in enumerate(users):
            client = Client(u, train_data[u], test_data[u])
            self.clients.append(client)
        
        return self.clients
    
    @abstractmethod
    def run(self):
        pass
    
    def save(self, path='./model.pt'):
        # for key in list(self.global_model.keys()):
        # self.global_model['module.' + key] = self.global_model.pop(key)
        self.model.set_params(self.global_model)
        self.model.save_model(path)
    
    def test(self, round, model=None):
        # Test model
        if round == 1 or round % self.eval_every == 0 or round == self.global_rounds:
            
            # metrics = self.train_proxy.test([client for client in self.clients], model)
            metrics = self.train_proxy.test([client for client in self.clients if not client.is_corrupted], model)
            metrics['round'] = round
            
            logging.info(metrics)
            if round >= 100 and metrics['accuracy'] < 30.0 and self.dataset == 'femnist':
                self.corrupted_counter += 1
                if self.corrupted_counter >= 3:
                    print("Broken down!")
                    exit()
            else:
                self.corrupted_counter = 0
            print('------Current round %d, time cost %.2f------' % (round, time.time() - self.time_begin), metrics)
