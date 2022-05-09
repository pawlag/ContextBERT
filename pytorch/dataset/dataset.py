from torch.utils.data import Dataset
import tqdm
import torch
import random
from util.patent_loader import load_documents_from_file, process_docs, measure_files, get_next_line_from_files
import os


class MultifileDataset(Dataset):
    def __init__(self, corpus_path, vocab, max_seq_len, max_tokens_len, min_seq_len, corpus_lines =0, context=False, context_vocab=None):
        self.vocab      = vocab        
        self.max_seq_len    = max_seq_len
        self.max_tokens_len = max_tokens_len
        self.min_seq_len    = min_seq_len
        self.context    = context
        self.context_vocab = context_vocab
        
        self.corpus_lines   = corpus_lines
        self.corpus_path    = corpus_path
        

        # count lines in corpus_lines == 0
        if self.corpus_lines == 0:
            _, self.corpus_lines = measure_files(self.corpus_path, self.min_seq_len, self.max_seq_len)
            print(f"total number of lines: {self.corpus_lines}")

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, idx):

        # get next context and subline
        context, subl =  next(get_next_line_from_files(self.corpus_path, self.min_seq_len, self.max_seq_len))

        # tokenize subl
        subl_encoded = self.vocab.tokenizer.encode(subl, is_pretokenized=True)
        
        # apply random mask and tokens switch and convert words to tokens ids
        ids_random, ids_label = self.perturbate_tokens(subl_encoded, subl)  

        # truncate ids lists if too long
        ids_random = ids_random[:self.max_tokens_len]
        ids_label = ids_label[:self.max_tokens_len]

        pad_idx = self.vocab.tokenizer.token_to_id("[PAD]")
        padding = [pad_idx]*(self.max_tokens_len - len(ids_random))
        ids_random.extend(padding)
        ids_label.extend(padding)

        output = {"bert_input": ids_random,
                  "bert_label": ids_label}

        
        if self.context:
            bert_context = [self.context_vocab.stoi.get(context, self.context_vocab.unk_index)]*len(ids_random)             
            c_padding = [self.context_vocab.pad_index]*len(padding) 
            bert_context.extend(c_padding)
            output["bert_context"]=bert_context

        return {key: torch.tensor(value) for key, value in output.items()}

    def perturbate_tokens(self, seq_encoded, seq):
        
        output_random = seq_encoded.ids
        output_label = seq_encoded.ids

        # iter through words in seq
        for i in range(len(seq)):
            prob = random.random()

            # perturbate 15% of word
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change id to mask id for whole word
                if prob < 0.8:
                    # get ids span for word
                    start, stop = seq_encoded.word_to_tokens(i) 
                    output_random[start:stop] = [self.vocab.tokenizer.token_to_id("[MASK]")]*(stop-start)

                # 10% randomly change token to random token from vocab
                elif prob < 0.9:
                    start, stop = seq_encoded.word_to_tokens(i) 
                    for idx in range(stop-start):
                        output_random[start+idx] = self.vocab.get_random_ids()

        return output_random, output_label

class BERTDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, context=False, context_vocab=None, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab      = vocab        
        self.seq_len    = seq_len
        self.context    = context
        self.context_vocab = context_vocab

        self.on_memory      = on_memory
        self.corpus_lines   = corpus_lines
        self.corpus_path    = corpus_path
        self.encoding       = encoding


        # sprawdz czy corpus_path to plik czy katalog
        # jak pliki to pobierz listę plików
        # ladowanie 

        with open(corpus_path, "r", encoding=encoding) as f:

            docs = load_documents_from_file(f.read())
            self.lines = process_docs(docs, max_length=seq_len-2, context=context, debug=False)   #  max_length = seq_len-2 aby zostawic miejsce na tokeny [CLS] i [SEP]        
            self.corpus_lines = len(self.lines)
            
        #     if self.corpus_lines is None and not on_memory:
        #         for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
        #             self.corpus_lines += 1

        #     if on_memory:
        #         self.lines = [line[:-1].split("\t")
        #                       for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
        #         self.corpus_lines = len(self.lines)

        # if not on_memory:
        #     self.file = open(corpus_path, "r", encoding=encoding)
        #     self.random_file = open(corpus_path, "r", encoding=encoding)

        #     for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
        #         self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        if self.context:
            sentence, context = self.lines[item]["text"], self.lines[item]["class"]
        else:
            sentence= self.lines[item]["text"]

        t_random, t_label = self.random_word(sentence)  

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t = [self.vocab.sos_index] + t_random + [self.vocab.eos_index]        
        t_label = [self.vocab.pad_index] + t_label + [self.vocab.pad_index]

        bert_input = t[:self.seq_len]
        bert_label = t_label [:self.seq_len]

        padding = [self.vocab.pad_index]*(self.seq_len - len(bert_input))
        bert_input.extend(padding)
        bert_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label}

        if self.context:
            bert_context = [self.context_vocab.pad_index] \
                        + [self.context_vocab.stoi.get(context, self.context_vocab.unk_index)]*len(t_random) \
                        + [self.context_vocab.pad_index]
            bert_context = bert_context[:self.seq_len]
            padding = [self.vocab.pad_index]* (self.seq_len - len(bert_context))
            bert_context.extend(padding)
            output["bert_context"]=bert_context

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)

        return tokens, output_label