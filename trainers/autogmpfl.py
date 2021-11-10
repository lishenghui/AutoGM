import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import linalg as LA

from models.fedbase import BaseClient
from models.fedmodel import FedModel
from .autogmfl import AutogmflServer
from .ditto import DittoModel


class AutogmpflModel(DittoModel):
    def __init__(self, net, seed, num_classes, lr, test_bs):
        super(AutogmpflModel, self).__init__(net, seed, num_classes, lr, test_bs)


class AutogmpflClient(BaseClient):
    def __init__(self, client_id, train_data, test_data, seed=0):
        super(AutogmpflClient, self).__init__(client_id, train_data, test_data, seed)


class AutogmpflServer(AutogmflServer):
    
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
                self.global_model[k] += avg_updates[k]
            return self.global_model
        else:
            return avg_updates
    
    def geometric_median_objective(self, median, points, alphas):
        """Compute geometric median objective."""
        return sum([alpha * self.l2dist(median, p) for alpha, p in zip(alphas, points)])
    
    def aggregate_rfa(self, param_list, clients, maxiter=10, eps=1e-5, ftol=1e-6):
        """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
        """
        # points = [client_model for (_, client_model, _) in self.updates]
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
                [alpha / max(eps, self.l2dist(median, p)) for alpha, p in zip(nor_weights, param_list)],
                dtype=nor_weights.dtype)
            weights = weights / weights.sum()
            median = self.aggregate(param_list, clients, weights, inplace=False)
            num_oracle_calls += 1
            obj_val = self.geometric_median_objective(median, param_list, nor_weights)
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                break
        
        for k in self.global_model.keys():
            self.global_model[k] += median[k]
        # self.global_model = median
        return self.global_model
    
    def l2dist(self, model1, model2):
        """L2 distance between p1, p2, each of which is a list of nd-arrays"""
        return LA.norm(torch.tensor([LA.norm(model1[k] - model2[k]) for k in model1]))
    
    def run(self):
        for client in self.clients:
            client.local_model = copy.deepcopy(self.global_model)
            client.set_datasets()
        
        total_train_samples = np.sum([client.num_train_samples for client in self.clients])
        for client in self.clients:
            client.scale = len(self.clients) * client.num_train_samples / total_train_samples
        
        for i in range(1, self.global_rounds + 1):
            if self.scale:
                selected_clients = self.select_clients(self.clients_per_round, weighted=False)
                updates = self.train_proxy.train(selected_clients, self.global_model, scale=True, lamb=self.lamb1,
                                                 init_model=self.init_global_model)
            else:
                selected_clients = self.select_clients(self.clients_per_round, weighted=True)
                updates = self.train_proxy.train(selected_clients, self.global_model, scale=False, lamb=self.lamb1,
                                                 init_model=self.init_global_model)
            
            local_models = [param[3] for param in updates]
            self.aggregate_autogm(local_models, selected_clients)
            for client, (_, _, local_model, _) in zip(selected_clients, updates):
                for k in local_model.keys():
                    client.local_model[k] = local_model[k]
            self.test(i)
