import torch
import torch.nn as nn
from  torch.nn.functional import one_hot



class ContextEncoding(nn.Module):
    """
    Context One Hot Encoder
        
    """

    def __init__(self, context_size):
        """
        :param context_size: total context vocab size        
        """
        super().__init__()
        self.size = context_size               
        

    def forward(self, sequence):

        return one_hot(sequence, num_classes=self.size).type(torch.float)


