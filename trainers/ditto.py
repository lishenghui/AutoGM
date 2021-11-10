import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from models.fedbase import BaseClient
from models.fedmodel import FedModel
from .autogmfl import AutogmflServer


class DittoModel(FedModel):
    def __init__(self, net, seed, num_classes, lr, test_bs):
        super(DittoModel, self).__init__(net, seed, num_classes, lr, test_bs)
    
    def update(self, dataset, epochs, g_net, l_net, noise, scale, **kwargs):
        curr_global_model = copy.deepcopy(g_net)
        curr_local_model = copy.deepcopy(l_net)
        if kwargs['free_rider']:
            return None, None, curr_local_model, curr_global_model
        self.net.train()
        self.set_params(curr_global_model)
        for data, target in dataset:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.net(data)
            loss = F.nll_loss(output, target) * scale
            loss.backward()
            self.optimizer.step()
        curr_global_model = copy.deepcopy(self.get_params())

        self.set_params(curr_local_model)
        for data, target in dataset:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.net(data)
            loss = F.nll_loss(output, target) * scale
            loss.backward()
            if kwargs['lamb'] > 0:
                for name, param in self.net.named_parameters():
                    if param.grad is not None:
                        param.grad += kwargs['lamb'] * (self.get_params()[name] - g_net[name])
            self.optimizer.step()
        curr_local_model = copy.deepcopy(self.get_params())
            
        if noise > 0:
            for k in curr_global_model.keys():
                curr_global_model[k] += torch.normal(noise, noise, size=curr_global_model[k].shape).to(self.device)
                curr_local_model[k] += torch.normal(noise, noise, size=curr_local_model[k].shape).to(self.device)
        
        return [None, None, curr_local_model, curr_global_model]


class DittoClient(BaseClient):
    def __init__(self, client_id, train_data, test_data, seed=0):
        super(DittoClient, self).__init__(client_id, train_data, test_data, seed)


class DittoServer(AutogmflServer):
    
    def __init__(self, params, net, dataset, client, model):
        super(AutogmflServer, self).__init__(params, net, dataset, client, model)
    
    def aggregate(self, update_list, clients, alphas=None, inplace=True):
        num_samples = np.array([client.num_train_samples for client in clients])
        
        if type(alphas) is np.ndarray:
            weights = alphas
        else:
            weights = num_samples / num_samples.sum()
        avg_updates = OrderedDict(
            [(k, torch.sum(torch.stack([param[k] * weight for param, weight in zip(update_list, weights)]), dim=0))
             for k in update_list[0]])
        if inplace:
            for k in self.global_model.keys():
                self.global_model[k] = avg_updates[k]
            return self.global_model
        else:
            return avg_updates
    
    def run(self):
        for client in self.clients:
            client.local_model = copy.deepcopy(self.global_model)
            client.set_datasets()
        
        total_train_samples = np.sum([client.num_train_samples for client in self.clients])
        for client in self.clients:
            client.scale = len(self.clients) * client.num_train_samples / total_train_samples
        
        for i in range(1, self.global_rounds + 1):
            # print(self.global_model)
            if self.scale:
                selected_clients = self.select_clients(self.clients_per_round, weighted=False)
                updates = self.train_proxy.train(selected_clients, self.global_model, scale=True, lamb=self.lamb1, init_model=self.init_global_model)
            else:
                selected_clients = self.select_clients(self.clients_per_round, weighted=True)
                updates = self.train_proxy.train(selected_clients, self.global_model, scale=False, lamb=self.lamb1, init_model=self.init_global_model)
            
            local_models = [param[3] for param in updates]
            self.aggregate(local_models, selected_clients)
            for client, (_, _, local_model, _) in zip(selected_clients, updates):
                for k in local_model.keys():
                    client.local_model[k] = local_model[k]
            # self.test(i, self.global_model)
            self.test(i)
