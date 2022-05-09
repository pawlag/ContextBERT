from calendar import c
import re
import json
from math import ceil
from collections import Counter
from random import shuffle
import os
import gzip
from datetime import datetime

rep_multispaces = re.compile(r'\s+')
rep_punct = re.compile(r'[><,:{})(;"`\'\"=—_/\\\+\[\]]')
rep_nonascii = re.compile(r'[^\x00-\x7f]')
max_dash_cnt = 3

def cleanzing(sentence):
    # sentence wise
    sen = rep_multispaces.sub(r' ', sentence) 
    sen = rep_punct.sub(r' ',sen) 
    sen = rep_nonascii.sub(r' ', sen)    
    
    # token wise
    tokens = [] 
    for token in sen.split():
        token_counter = Counter(token)
        if token_counter['-'] <= max_dash_cnt:
            if token_counter['-'] > 0:
                token = " ".join(token.split('-'))
            tokens.append(token) 
    sen = " ".join(tokens)
    return sen.strip()


def tokenize(sentence):
    return sentence.split()

def tokens_to_sentence(tokens):
    return " ".join(tokens)

def tokens_stats(tokens):
    return len(tokens), sum([len(t) for t in tokens])

def truncate(tokens, max_length, split = True):

    if not split:
        return [tokens[:max_length]]
    else:
        current_len = len(tokens)
        n = ceil(current_len/max_length)
        
        split_tokens = []
        for i in range(n):
            split_tokens.append(tokens[i*max_length:(i+1)*max_length])
        return  split_tokens

def load_documents_from_file(json_body):    

    docs = json.loads(json_body) 
    print(f"loaded n {len(docs)}")
        
    return docs


def  process_docs(documents, max_length=None, context = True, class_chars_n = 3, debug=False):    

    if debug: print(f"documents to process {len(documents)}")

    output = []

    for doc in documents:

        #   patent classification
        if context:

            classifications = doc.get('classifications')
            if debug:
                print(classifications)
            classes=set()
            if classifications:              
                for class_type in classifications:
                    if class_type in ["cpc", "ipc", 'ipcr']:
                        new_classes = set([code[:class_chars_n] for code in classifications[class_type]["codes"]])
                        classes=classes.union(new_classes)
            else:
                classes = None
            if debug: print(f"classes: {classes}")
        # patent text ['classifications']['cpc']['codes']
        doc_sentences = doc.get('text','').split('.')
        if debug: print(f"sentences in doc {len(doc_sentences)}")

        for sen in doc_sentences:        
            # extra cleansing on fly            
            sen = cleanzing(sen)
            # tokens
            tokens=tokenize(sen)
            # if debug: print(f"tokens: {tokens}")
            if max_length:
                list_of_tokens = truncate(tokens, max_length)
            else:
                list_of_tokens=[tokens]
            for tokens in list_of_tokens:
                tokens_cnt, chars_cnt = tokens_stats(tokens)                
                if tokens_cnt >= 5 and chars_cnt >= 15: 
                    if context:
                        for _class in classes:
                            output.append({"text":tokens_to_sentence(tokens),"class":_class})
                            if debug: print(f"class:{_class}")
                    else:
                        output.append({"text":tokens_to_sentence(tokens)})

    return output


def get_list_of_files(dir_name, randomize = True):
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

    # shuffle files
    if randomize:
        shuffle(files) 

    return files

def create_training_files(input_path, output_path, compress=True, max_files = None, overwrite = False):

    # get files
    files = get_list_of_files(input_path)
    if max_files:
        files = files[:max_files]


    for i, file in enumerate(files):

        if compress:
            out_file = os.path.join(output_path, str(os.path.split(os.path.splitext(file)[0])[1])+".txt.gz")
        else:
            out_file = os.path.join(output_path, str(os.path.split(os.path.splitext(file)[0])[1])+".txt")
        if overwrite or not os.path.exists(out_file):
            t = datetime.now()
            with open(file, "r", encoding='utf-8') as f:
                print(f"processing file {file}, {i+1}/{len(files)}")            
                docs = load_documents_from_file(f.read())
                docs = process_docs(docs, max_length=None, context = True, debug=False)
            
            if compress:
                out_file = os.path.join(output_path, str(os.path.split(os.path.splitext(file)[0])[1])+".txt.gz")
            else:
                out_file = os.path.join(output_path, str(os.path.split(os.path.splitext(file)[0])[1])+".txt")
            print(f"writing to file {out_file}")

            # save to output
            with gzip.open(out_file,'wt') if compress else open(out_file,'w') as f:
                for doc in docs:
                    f.write(f"{doc['class']}|{doc['text']}\n")
            print(f"time {datetime.now()-t}")

        else:
            print(f"output file {out_file} already exists, {i+1}/{len(files)}")            



def measure_files(input_path, min_seq_len, max_seq_len, max_word_len, tokenizer = None):

    min_line =  max_seq_len
    max_line = 0
    max_tokens = 0
    max_tokens_sequences = None
    tokens_to_lines_ratio = 0
    context_counter = Counter()
    line_counter = 0 
    for context, line in get_next_line_from_files(input_path, min_seq_len, max_seq_len, max_word_len):
        context_counter[context]+=1
        line_counter+=1
        ll = len(line)
        if ll > max_line: max_line=ll
        if ll < min_line: min_line=ll

        if tokenizer:
            tokenizer_out = tokenizer.encode(line, is_pretokenized=True)
            l_tokens = len(tokenizer_out.tokens)
            tokens_to_lines_ratio = (tokens_to_lines_ratio*(line_counter-1) + (l_tokens/ll))/line_counter
            if  l_tokens > max_tokens:
                max_tokens = l_tokens
                max_tokens_sequences = {"line":line, "tokens":tokenizer_out.tokens}
            

        if line_counter % 1e6 == 0:
            print("context_counter size: ", len(context_counter))
            print("line_counter: ", int(line_counter/1e6),"M")            
            print("min, max line: ", min_line, max_line)
            print("max tokens: ", max_tokens)
            print("tokens to lines ratio: ", tokens_to_lines_ratio)
            #print("max tokens case: ", max_tokens_sequences)
            #print("line: ", line)

    print("context_counter: ", context_counter)
    print("context_counter size: ", len(context_counter))
    print("line_counter: ", int(line_counter/1e6),"M")            
    print("min, max line: ", min_line, max_line)
    print("max tokens: ", max_tokens)
    print("tokens to lines ratio: ", tokens_to_lines_ratio)
    print("max tokens case: ", max_tokens_sequences)

    return context_counter, line_counter

def get_next_line_from_files(input_path, min_seq_len, max_seq_len, max_word_len):

    context_right_patter = re.compile(r'[A-Z][0-9]{2}')

    for corpus_file in get_list_of_files(input_path):
        with open(corpus_file, "r") as f:
            for line in f:
                # split into context and sequence
                context_seq = line.split('|')
                if len(context_seq) == 2:

                    context, seq = context_seq

                    # clean context
                    context = context[0:3].upper()
                    # check pattern
                    if not context_right_patter.match(context):
                        context='[UNK]'

                    # split into words and filter out too long words
                    tokens = [word for word in seq.split() if len(word) <= max_word_len]
                    
                    n_tokens = len(tokens) 
                    if n_tokens >= min_seq_len:
                        n_sublines = int(n_tokens / max_seq_len)
                        if n_tokens % max_seq_len >= min_seq_len:
                            n_sublines +=1
                        for i in range(n_sublines):                            
                            yield context, tokens[i*max_seq_len : (i+1)*max_seq_len]


if __name__ == '__main__':
    
    files_path = '/mnt/corpus_files/'    
    context_counter, line_counter = measure_files(files_path, 3, 20, 50)
    print(f"context_counter {context_counter}")
    print(f"line_counter {line_counter}")   


  