import torch.nn as nn

from .transformer import  ContextTransformerBlock
from .embedding import BERTEmbedding, ContextEmbedding
from .utils import FeedForward


class ContextBERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers with context as global state
    """

    def __init__(self, vocab_size, context_size, hidden=256, n_layers=6, attn_heads=6, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.vocab_size = vocab_size
        self.context_size = context_size

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4
        self.context_feed_forward_hidden = hidden * 2

        # embedding for BERT, sum of positional, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # embedding for context inputs (sequence-wise)
        self.context_embedding = ContextEmbedding(context_size=context_size, embed_size=hidden)

        # fnn for context
        self.context_fnn = FeedForward(d_model=hidden, d_ff=self.context_feed_forward_hidden, dropout=dropout)
        # multi-layers transformer blocks attendig to context state
        self.transformer_blocks = nn.ModuleList(
            [ContextTransformerBlock(hidden, attn_heads, self.feed_forward_hidden, dropout) for _ in range(n_layers)])

    def forward(self, x, c):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)        
        c = self.context_embedding(c)

        # context through fnn
        c = self.context_fnn(c)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, c, mask)

        return x