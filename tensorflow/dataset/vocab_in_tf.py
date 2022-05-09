from calendar import c
import collections
import sys
import os.path as path
import pickle
from random import sample, shuffle
from math import ceil
from datetime import datetime
from typing import Counter

from tensorflow_text.tools.wordpiece_vocab.wordpiece_tokenizer_learner_lib import learn

sys.path.append('/home/luser/repos/ContextBERT')
from util.patent_loader import load_documents_from_file, process_docs, get_list_of_files


def count_words_in_files(input_path, output_path, prob=None):

    files = get_list_of_files(input_path)

    for i, file in enumerate(files):
            
        with open(file, "r", encoding='utf-8') as f:
            print(f"processing file {file}, {i+1}/{len(files)}")
            t = datetime.now()
            docs = load_documents_from_file(f.read())
            docs = process_docs(docs, max_length=None, context = False, debug=False)

            # new counter for file
            word_counter = collections.Counter()

            for doc in docs:
                txt = doc["text"].split()                
                if prob:
                    l = len(txt)
                    txt = sample(txt, min(ceil(prob*l),l))
                
                word_counter.update(txt)

            # save counter for file
            print(f"\tunique word cnt {len(word_counter)}")
            counter_file = path.join(output_path, str(path.split(path.splitext(file)[0])[1])+".pck")
            with open(counter_file,'wb') as cf:
                pickle.dump(word_counter,cf)
            print(f"\ttime {datetime.now()-t}")

def merge_counters(input_path):

    files = get_list_of_files(input_path)

    counter = Counter()

    for i, file in enumerate(files):
        print(f"loading file {file}, {i+1}/{len(files)}")
        with open(file, 'rb') as f:
            _counter = pickle.load(f)
            counter.update(_counter)
            
    return counter



if __name__ == "__main__":

    input_path = '/mnt/txt_base/files/'
    output_path = '/home/luser/data/patent/counters/'
    counter_pck_f = '/home/luser/data/patent/model/wp_counter.pck'
    word_to_learn_f = '/home/luser/data/patent/model/word_to_learn.txt'

    #count_words_in_files(input_path, output_path)

    if path.exists(counter_pck_f):
        with open(counter_pck_f, 'rb') as f:
            counter = pickle.load(f)
    else:
        counter = merge_counters(output_path)
        # save
        with open(counter_pck_f, 'wb') as f:
            pickle.dump(counter, f)

    print("total vocab size ",len(counter))


    # cleaning of preliminary vocab       
    min_cnt = 10
    max_word_len = 50    

    words_to_learn = []
    i = 0
    for word, cnt in counter.items():
        if cnt >= min_cnt and len(word) <= max_word_len:
            words_to_learn.append((word,cnt)) 
            i+=1
            if i % 100 == 0:
                print(word)
    
    print(f"vocab to learn, freq min {min_cnt} ", len(words_to_learn))
    # save
    with open(word_to_learn_f, 'w') as f:
        for word, cnt in words_to_learn:
            f.write(f"{word}\t{cnt}\n")
        

    wp_vocab = learn(words_to_learn, 75000, reserved_tokens= ['[PAD]', '[UNK]', "[CLS]", "[SEP]", "[MASK]"], max_input_tokens=-1)

    print("wp vocab size ",len(wp_vocab))
            
    # save
    with open('/home/luser/data/patent/model/wp_vocab.pck', 'wb') as f:
        pickle.dump(wp_vocab,f)
    
    with open('/home/luser/data/patent/model/wp_vocab.txt', 'w') as f:
        for i, t in enumerate(wp_vocab):
            f.write(f'"{t}":{i},\n')