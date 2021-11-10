import copy
from collections import OrderedDict

import numpy as np
import torch

from models.fedbase import BaseClient, BaseServer
from models.fedmodel import FedModel


class FedavgModel(FedModel):
    def __init__(self, net, seed, num_classes, lr, test_bs):
        super(FedavgModel, self).__init__(net, seed, num_classes, lr, test_bs)


class FedavgClient(BaseClient):
    def __init__(self, client_id, train_data, test_data, seed=0):
        super(FedavgClient, self).__init__(client_id, train_data, test_data, seed)


class FedavgServer(BaseServer):
    
    def __init__(self, params, net, dataset, client, model):
        super(FedavgServer, self).__init__(params, net, dataset, client, model)
    
    def aggregate(self, update_list, clients, alphas=None, inplace=True):
        num_samples = np.array([client.num_train_samples for client in clients])
        weights = num_samples / num_samples.sum()
        nor_weights = np.array(weights) / np.sum(weights)
        avg_updates = OrderedDict(
            [(k, torch.sum(torch.stack([param[k] * weight for param, weight in zip(update_list, nor_weights)]), dim=0))
             for k in update_list[0]])
        if inplace:
            self.global_model = avg_updates
            return self.global_model
        return avg_updates
    
    def run(self):
        for client in self.clients:
            client.local_model = copy.deepcopy(self.global_model)
            client.set_datasets()
        
        for i in range(1, self.global_rounds + 1):
            selected_clients = self.select_clients(self.clients_per_round, weighted=False)
            updates = self.train_proxy.train(selected_clients, self.global_model, scale=False, init_model=self.init_global_model)
            
            local_models = [param[3] for param in updates]
            self.aggregate(local_models, selected_clients)
            
            self.test(i, self.global_model)
