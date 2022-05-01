from calendar import c
import re
import json
from math import ceil
from collections import Counter
from random import shuffle
import os


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


if __name__ == '__main__':

    with open('/mnt/txt_base/files/ipa140710.json','r') as f:

        docs = load_documents_from_file(f.read())
        process_docs(docs, max_length=20, debug = True)      

  