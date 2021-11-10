import copy
from collections import OrderedDict

import numpy as np
import torch
from torch import linalg as LA

from models.fedbase import BaseClient, BaseServer
from models.fedmodel import FedModel


class AutogmflModel(FedModel):
    def __init__(self, net, seed, num_classes, lr, test_bs):
        super(AutogmflModel, self).__init__(net, seed, num_classes, lr, test_bs)


class AutogmflClient(BaseClient):
    def __init__(self, client_id, train_data, test_data, seed=0):
        super(AutogmflClient, self).__init__(client_id, train_data, test_data, seed)
        self.__alpha = 1
        self.__distance = 1
    
    @property
    def distance(self):
        return self.__distance
    
    @distance.setter
    def distance(self, distance):
        self.__distance = distance
    
    @property
    def alpha(self):
        return self.__alpha
    
    @alpha.setter
    def alpha(self, alpha):
        self.__alpha = alpha


class AutogmflServer(BaseServer):
    
    def __init__(self, params, net, dataset, client, model, num_classes=10):
        super(AutogmflServer, self).__init__(params, net, dataset, client, model, num_class=num_classes)
        self.weights = np.ones(shape=len(self.clients)) / len(self.clients)
    
    def geometric_median_objective(self, median, points, alphas):
        return sum([alpha * self.l2dist(median, p) for alpha, p in zip(alphas, points)])
    
    def global_objective(self):
        pass
    
    def weighted_average_oracle(self, points, weights):
        tot_weights = np.sum(weights)
        weighted_updates = [np.zeros_like(v) for v in points[0]]
        
        for w, p in zip(weights, points):
            for j, weighted_val in enumerate(weighted_updates):
                weighted_val += (w / tot_weights) * p[j]
        
        return weighted_updates
    
    def aggregate_autogm(self, param_list, clients, maxiter=10, eps=1e-4, ftol=1e-6):
        """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
        """
        lamb = self.lamb2 * len(clients)
        # lamb = self.lamb2 * sum([client.num_train_samples for client in clients])
        alpha = np.ones(len(clients)) / len(clients)
        for i in range(maxiter):
            median = self.aggregate(param_list, clients, alpha, inplace=False)
            obj_val = self.geometric_median_objective(median, param_list, alpha)
            global_obj = obj_val + lamb * np.linalg.norm(alpha) ** 2 / 2
            for i in range(maxiter):
                prev_median, prev_obj_val = median, obj_val
                weights = np.asarray(
                    [alpha_i / max(eps, self.l2dist(median, p)) for alpha_i, p in zip(alpha, param_list)],
                    dtype=alpha.dtype)
                weights = weights / weights.sum()
                median = self.aggregate(param_list, clients, weights, inplace=False)
                obj_val = self.geometric_median_objective(median, param_list, alpha)
                if abs(prev_obj_val - obj_val) < ftol * obj_val:
                    break
            
            self.global_model = median
            for client, local_model in zip(clients, param_list):
                client.distance = self.distance2global(local_model)
            
            idxs = [x for x, _ in sorted(enumerate(clients), key=lambda x: x[1].distance)]
            eta_optimal = 10000000000000000.0  # clients[idxs[0]].distance + lamb
            for p in range(0, len(idxs)):
                eta = (sum([clients[i].distance for i in idxs[:p + 1]]) + lamb) / (p + 1)
                if p < len(idxs) and eta - clients[idxs[p]].distance < 0:
                    break
                else:
                    eta_optimal = eta
            alpha = np.array([max(eta_optimal - c.distance, 0) / lamb for c in
                              clients])
            
            new_obj = obj_val + lamb * np.linalg.norm(alpha) ** 2 / 2
            if abs(new_obj - global_obj) < ftol * new_obj:
                break
    
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
    
    def distance2global(self, model, lamb=1):
        """L2 distance between p1, p2, each of which is a list of nd-arrays"""
        dis = LA.norm(torch.tensor([LA.norm(self.global_model[k] - model[k]) for k in self.global_model])).item()
        if not dis:
            dis = 0.0000001
        return dis
    
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
                updates = self.train_proxy.train(selected_clients, self.global_model, scale=True, init_model=self.init_global_model)
            else:
                selected_clients = self.select_clients(self.clients_per_round, weighted=True)
                updates = self.train_proxy.train(selected_clients, self.global_model, scale=False, init_model=self.init_global_model)
            
            local_models = [param[3] for param in updates]
            self.aggregate_autogm(local_models, selected_clients)
            self.test(i, self.global_model)
