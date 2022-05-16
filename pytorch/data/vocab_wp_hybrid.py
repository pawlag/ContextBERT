from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from random import sample, shuffle, randrange
from math import ceil
import pandas as pd
from collections import Counter 
from util.patent_loader import load_documents_from_file, process_docs, get_list_of_files

class WordPieceVocab():
    """
    Defines a vocabulary using WordPiece algorithm implementation in Huggingface
    """
    def __init__(self):
        self.tokenizer = Tokenizer(WordPiece(unk_token='[UNK]'))
        self.tokenizer.pre_tokenizer = Whitespace()
      
    def load(self, path):
        self.tokenizer = self.tokenizer.from_file(path)
    
    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def get_random_ids(self):
        return  randrange(self.tokenizer.get_vocab_size())

if __name__ == "__main__":
    vb = WordPieceVocab()
    vb.load('/home/luser/data/patent/model/wp_vocab_hybrid_freq10k_75k.json')
    print(type(vb.tokenizer))
    # check on sample abstract
    txt = '''
    The invention relates to a dual-function tool, i.e. having two interchangeable tools, one providing a shovel function and the other providing a fork function, both suitable for digging the ground. For this purpose, the tool comprises a shaft (<b>1</b>) with one end provided with a cross-piece to form a cross-handle, and the other end provided with means for mounting a digging tool (<b>3</b>-<b>3&#x2032;</b>), such as a shovel or fork, such that the corresponding shaft (<b>1</b>) comprises at least two sections (<b>1&#x2032;</b>-<b>1&#x2033;</b>) which can be removably screwed to one another, the cross-piece of the upper handle comprising a T-shaped connector (<b>5</b>) which can be removably coupled on the upper end of the shaft, and two side elements (<b>7</b>) which can be screwed onto the end of each of the side arms of the T-shaped connector, together forming an upper cross-piece having an oversized length. The interchangeable tool is mounted on the lower end of the shaft
    '''
    pre_tokens = txt.lower().replace("-"," ").replace("."," ").replace(","," ").split()
    
    print(len(pre_tokens))
    print(pre_tokens)
    out = vb.tokenizer.encode(pre_tokens, is_pretokenized=True)

    print(len(out.tokens))
    print(out.tokens)
    print(out.ids)
    print(out.word_to_tokens(5))

    print(f"[UNK] id {vb.tokenizer.token_to_id('[UNK]')}")
    

    # # check on training data
    # patent_train_path = '/home/luser/data/patent/us-patent-phrase-to-phrase-matching/train.csv'

    # train_data = pd.read_csv(patent_train_path)
    # print(train_data.size)

    # train_data_processed= []
    # for i in range(train_data.shape[0]):
    #     context = str(train_data["context"][i])
    #     anchor = str(train_data["anchor"][i])
    #     anchor_pretokens = anchor.split()
    #     anchor_tokens = vb.tokenizer.encode(anchor_pretokens, is_pretokenized=True)
    #     target = str(train_data["target"][i])
    #     target_pretokens = target.split()
    #     target_tokens = vb.tokenizer.encode(target_pretokens, is_pretokenized=True)
    #     train_data_processed.append({"context": context, 
    #                                "anchor":anchor, 
    #                                "target":target, 
    #                                "anchor_len":len(anchor_pretokens),
    #                                "target_len":len(target_pretokens),
    #                                "anchor_tokens":anchor_tokens.tokens,
    #                                "target_tokens":target_tokens.tokens,                                   
    #                                "anchor_tokens_len":len(anchor_tokens.tokens),
    #                                "target_tokens_len":len(target_tokens.tokens),
    #                                "anchor_unkown_token":Counter(anchor_tokens.tokens)["[UNK]"],
    #                                "target_unkown_token":Counter(target_tokens.tokens)["[UNK]"]
    #                                 })

    # train_data_processed = pd.DataFrame(train_data_processed)
    # print(train_data_processed)

    # unknow_mask = train_data_processed[(train_data_processed["anchor_unkown_token"] > 0) | (train_data_processed["target_unkown_token"] > 0)]
    # print(unknow_mask)

    # unknow_mask.to_csv("/home/luser/data/patent/us-patent-phrase-to-phrase-matching/train_tokenization_unkown.csv")