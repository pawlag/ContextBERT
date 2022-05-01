import torch.nn as nn
from .bert import BERT
from .contextual_bert import ContextBERT
from typing import Union


class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: Union[BERT, ContextBERT]):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()

        self.bert = bert        

        if type(bert).__name__ == 'ContextBERT':
            self.context_aware = True
        else:
            self.context_aware = False

        self.mask_lm = MaskedLanguageModel(self.bert.hidden, self.bert.vocab_size)

    def forward(self, x, c=None):
        
        x = self.bert(x,c) if self.context_aware else self.bert(x)

        return self.mask_lm(x)



class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
