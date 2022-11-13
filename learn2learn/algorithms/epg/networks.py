import torch 
import torch.nn.functional as F
import traceback
from torch.autograd import grad
from torch import optim

###pytorch version 
def linear_init(module):
        if isinstance(module, F.Linear):
            F.init.xavier_uniform_(module.weight)
            module.bias.data.zero_()
        return module

class Memory(object):
    """Linear memory, allows storing information in its weights.
    """
    def __init__(self, input_size, output_size, hiddens=None, activation='tanh', device='cpu'):
        super(Memory, self).__init__()
        self.device = device
        if hiddens is None:
            hiddens = [64]
        if activation == 'relu':
            activation = F.ReLU
        elif activation == 'tanh':
            activation = F.Tanh
        layers = [linear_init(F.Linear(input_size, hiddens[0])), activation()]
        for i, o in zip(hiddens[:-1], hiddens[1:]):
            layers.append(linear_init(F.Linear(i, o)))
            layers.append(activation())
        layers.append(linear_init(F.Linear(hiddens[-1], output_size)))
        self.memory = F.Sequential(*layers)
        
    def forward(self,x):
         input = x
         memory = self.memory(x)
         return memory 
        
       
