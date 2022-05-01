import pickle
import tqdm
import os
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from random import sample, shuffle
from math import ceil


from util.patent_loader import load_documents_from_file, process_docs

class WordPieceVocab():
    """Defines a vocabulary using WordPiece algoritm implementation in Huggingface
    """

    def __init__(self):
        self.tokenizer = Tokenizer(WordPiece(unk_token='[UNK]'))
        self.tokenizer.pre_tokenizer = Whitespace()
    

    def train(self, 
                corpus, 
                from_files = False,
                max_size=50000, 
                min_freq=0, 
                special_tokens=['[PAD]', '[UNK]', "[CLS]", "[SEP]", "[MASK]"]):

        _trainer = WordPieceTrainer(vocab_size = max_size, min_frequency=min_freq, special_tokens=special_tokens)

        if from_files:
            self.tokenizer.train(corpus, _trainer)
        else:
            self.tokenizer.train_from_iterator(corpus, _trainer)  

        print("Vocab size:", self.tokenizer.get_vocab_size())    
    
    def save(self, path):
        self.tokenizer.save(path)
    
    def load(self, path):
        self.tokenizer = self.tokenizer.from_file(path)
    
    def __len__(self):       

        return self.tokenizer.get_vocab_size()

def get_list_of_files(dir_name):
    # create a list of file and sub directories 
    # names in the given directory 
    obj_in_path = os.listdir(dir_name)
    files = list()
    # Iterate over all the entries
    for obj in obj_in_path:
        # Create full path
        full_path = os.path.join(dir_name, obj)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(full_path):
            files = files + get_list_of_files(full_path)
        else:
            files.append(full_path)
            
    return files

def process_files_as_generator(files, encoding, prob = None):

    for i, file in enumerate(files):
        with open(file, "r", encoding=encoding) as f:
            print(f"processing file {file}, {i}")
            docs = load_documents_from_file(f.read())
            docs = process_docs(docs, max_length=None, context = False, debug=False)
            if prob:
                out = []
                for doc in docs:
                    txt = doc["text"].split()
                    l = len(txt)
                    txt = " ".join(sample(txt, min(ceil(prob*l),l)))                    
                    out.append(txt)
                yield out

            yield [doc["text"] for doc in docs]



def build():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corpus_path",  required=True, type=str)
    parser.add_argument("-o", "--output_path",  required=True, type=str)
    parser.add_argument("-n", "--n_files",      type=int, default=52)
    parser.add_argument("-s", "--vocab_size",   type=int, default=None)
    parser.add_argument("-e", "--encoding",     type=str, default="utf-8")
    parser.add_argument("-m", "--min_freq",     type=int, default=1)
    args = parser.parse_args()


    root_path = args.corpus_path

    if os.path.isdir(root_path): # root_path is a dir, process all files in this dir at its subdirectories
        vocab = WordPieceVocab()
        files = get_list_of_files(root_path)
        shuffle(files)
        files = files[:args.n_files]        
        #vocab.train(process_files_as_generator(files, encoding=args.encoding, prob = 0.2))
        vocab.train(process_files_as_generator(files, encoding=args.encoding))
        vocab.save(args.output_path)
    
    else: # root_path is a file
        with open(root_path, "r", encoding=args.encoding) as f:
            docs = load_documents_from_file(f.read())
            docs = process_docs(docs, max_length=None, context = False, debug=False)
            sentences = [doc["text"] for doc in docs]
            vocab = WordPieceVocab()
            vocab.train(sentences)
            vocab.save(args.output_path)


if __name__ == '__main__':
    build()

    # python vocab_from_wordpiece.py -c /mnt/txt_base/files/ipa140710.json -o /home/luser/data/patent/model/wp_vocab.json
    # 
    # 
    # python vocab_from_wordpiece.py -c /mnt/txt_base/files -o /home/luser/data/patent/model/wp_vocab.json 
