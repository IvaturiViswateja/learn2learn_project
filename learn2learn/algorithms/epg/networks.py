import torch 
import torch.nn.functional as F
import traceback
from torch.autograd import grad

from torch import optim





#convert this to pytorch linear model
class Memory(object):
    """Linear memory, allows storing information in its weights.
    """

    def __init__(self, size_in=None, size_out=None):
        assert size_in is not None
        assert size_out is not None

        shp = size_out, size_in
        # Correctly init weights.
        out = np.random.randn(*shp).astype(np.float32)
        out *= 0.01 / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        self._w = C.Variable(out)
        self._b = C.Variable(np.zeros(shp[0], dtype=np.float32))
        self.train_vars = [self._w, self._b]

    def f(self):
        x = np.ones(self._w.shape[1], dtype=np.float32)[np.newaxis, :]
        # Needs nonlinearity after memory
        return F.tanh(F.linear(x, self._w, self._b))
class Memory(object):
    """Linear memory, allows storing information in its weights.
    """
    def __init__(self, input_size, output_size, hiddens=None, activation='tanh', device='cpu'):
        super(Memory, self).__init__()
        self.device = device
        if hiddens is None:
            hiddens = [64]
        if activation == 'relu':
            activation = nn.ReLU
        elif activation == 'tanh':
            activation = nn.Tanh
        layers = [linear_init(nn.Linear(input_size, hiddens[0])), activation()]
        for i, o in zip(hiddens[:-1], hiddens[1:]):
            layers.append(linear_init(nn.Linear(i, o)))
            layers.append(activation())
        layers.append(linear_init(nn.Linear(hiddens[-1], output_size)))
        self.memory = nn.Sequential(*layers)
        
     def forward(self,x)
         input = x
         memory = self.memory(x)
         return memory 
        
       
