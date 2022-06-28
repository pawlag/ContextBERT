import torch.nn as nn



class ContextEmbedding(nn.Module):
    """
    Context Embedding 
        
    """

    def __init__(self, context_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.context_size = context_size       
        self.embed = nn.Embedding(context_size, embed_size, padding_idx=0)        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence):
        x = self.embed(sequence) 
        return self.dropout(x)
