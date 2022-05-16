from dataset import MultifileDataset
from vocab_wp_hybrid import WordPieceVocab
from context_vocab import Vocab

#data = MultifileDataset(corpus_path, vocab, max_seq_len, max_tokens_len, min_seq_len, max_word_len, corpus_lines =0, context=False, context_vocab=None)
vb = WordPieceVocab()
vb.load('/home/luser/data/patent/model/wp_vocab_hybrid_freq10k_75k.json')
cvb = Vocab().load('/home/luser/data/patent/model/context_vb_fixed.pck')


# data = MultifileDataset('/mnt/corpus_files_sample/', 
#                         vb, 
#                         20, 25, 3, 50, corpus_lines =4529557, context=True, context_vocab=cvb)

# for i in range(10):
#     for item in data[i].items():
#         print(item)
