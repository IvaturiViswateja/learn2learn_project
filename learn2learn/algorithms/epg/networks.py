import torch 
import torch.nn.functional as F
import traceback
from torch.autograd import grad

from torch import optim



import chainer as C
import chainer.functions as F
import numpy as np


    def set_params_1d(self, params):
        """Set params for ES (theta)
        """
        n = self._lst_w + self._lst_b
        idx = 0
        for e in n:
            e.data[...] = params[idx:idx + e.size].reshape(e.shape)
            idx += e.size

    def get_params_1d(self):
        """Get params for ES (theta)
        """
        n = self._lst_w + self._lst_b
        return np.concatenate([e.data.flatten() for e in n])


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
