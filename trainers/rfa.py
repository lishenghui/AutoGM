import copy
from collections import OrderedDict

import numpy as np
import torch
from torch import linalg as LA

from models.fedbase import BaseClient, BaseServer
from models.fedmodel import FedModel


class RfaModel(FedModel):
    def __init__(self, net, seed, num_classes, lr, test_bs):
        super(RfaModel, self).__init__(net, seed, num_classes, lr, test_bs)
    
    def l2dist(self, model1, model2):
        """L2 distance between p1, p2, each of which is a list of nd-a®rrays"""
        return LA.norm(torch.tensor([LA.norm(model1[k] - model2[k]) for k in model1]))


class RfaClient(BaseClient):
    def __init__(self, client_id, train_data, test_data, seed=0):
        super(RfaClient, self).__init__(client_id, train_data, test_data, seed)


class RfaServer(BaseServer):
    
    def __init__(self, params, net, dataset, client, model):
        super(RfaServer, self).__init__(params, net, dataset, client, model)
    
    def aggregate(self, update_list, clients, alphas=None, inplace=True):
        if alphas is not None:
            weights = alphas
        else:
            weights = np.array([client.alpha for client in clients])
        nor_weights = np.array(weights) / np.sum(weights)
        avg_updates = OrderedDict(
            [(k, torch.sum(torch.stack([param[k] * weight for param, weight in zip(update_list, nor_weights)]), dim=0))
             for k in update_list[0]])
        if inplace:
            self.global_model = avg_updates
            return self.global_model
        return avg_updates
    
    def geometric_median_objective(self, median, points, alphas):
        """Compute geometric median objective."""
        return sum([alpha * self.l2dist(median, p) for alpha, p in zip(alphas, points)])
    
    def aggregate_rfa(self, param_list, clients, maxiter=10, eps=1e-4, ftol=1e-6):
        """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
        """
        # weights = np.ones(len(clients)) / len(clients)
        num_samples = np.array([client.num_train_samples for client in clients])
        weights = num_samples / num_samples.sum()
        nor_weights = np.array(weights) / np.sum(weights)
        median = self.aggregate(param_list, clients, nor_weights, inplace=False)
        num_oracle_calls = 1
        obj_val = self.geometric_median_objective(median, param_list, nor_weights)
        
        # start
        for i in range(maxiter):
            prev_median, prev_obj_val = median, obj_val
            weights = np.asarray(
                [max(eps, alpha / max(eps, self.l2dist(median, p))) for alpha, p in zip(nor_weights, param_list)],
                dtype=nor_weights.dtype)
            weights = weights / weights.sum()
            median = self.aggregate(param_list, clients, weights, inplace=False)
            num_oracle_calls += 1
            obj_val = self.geometric_median_objective(median, param_list, nor_weights)
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                break
        # print(weights)
        self.global_model = median
        return self.global_model
    
    def l2dist(self, model1, model2):
        """L2 distance between p1, p2, each of which is a list of nd-arrays"""
        return LA.norm(torch.tensor([LA.norm(model1[k] - model2[k]) for k in model1]))
    
    def run(self):
        for client in self.clients:
            client.local_model = copy.deepcopy(self.global_model)
            client.set_datasets()
        
        for i in range(1, self.global_rounds + 1):
            selected_clients = self.select_clients(self.clients_per_round, weighted=False)
            updates = self.train_proxy.train(selected_clients, self.global_model, scale=False, init_model=self.init_global_model)
            
            self.aggregate_rfa([param[3] for param in updates], selected_clients)
            self.test(i, self.global_model)
