import copy

import torch
import torch.nn.functional as F
import torch.optim as optim


class FedModel():
    def __init__(self, net, seed, num_classes, lr, test_bs=128):
        self.lr = lr
        self.test_bs = test_bs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_random_seed(seed)
        self.net = net(num_classes).to(self.device)
        # if torch.cuda.device_count() > 1:
        #     self.net = nn.DataParallel(self.net)
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr)
        
    
    def set_params(self, model_params):
        self.net.load_state_dict(copy.deepcopy(model_params))
    
    def save_model(self, path='./'):
        torch.save(self.net, path)
    
    def get_params(self):
        return self.net.state_dict()
    
    def set_random_seed(self, seed_value):
        torch.manual_seed(seed_value)  # cpu  vars
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)  # gpu vars
            torch.backends.cudnn.deterministic = True  # needed
            torch.backends.cudnn.benchmark = False
    
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
        
        if noise > 0:
            for k in curr_global_model.keys():
                curr_global_model[k] += torch.normal(noise, noise, size=curr_global_model[k].shape).to(self.device)
        return [None, None, None, curr_global_model]
    
    def test_model(self, test_loader):
        return self.net.test_model(test_loader, self.device)
