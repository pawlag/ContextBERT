from regex import P
import yaml
import argparse
from collections import namedtuple

from torch.utils.data import DataLoader
from model.bert import BERT
from model.contextual_bert import ContextBERT
from model.trainer import BERTTrainer
from data import MultifileDataset, WordPieceVocab
from data.context_vocab import Vocab


def to_nametuple(dict_data):
    return namedtuple("Config", dict_data.keys())(*tuple(map(lambda x: x if not isinstance(x, dict) else to_nametuple(x), dict_data.values())))


def train():

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str, help="path to config file")
    args = parser.parse_args()

    # load config
    with open(args.config, "r") as conf_f:
        conf = yaml.safe_load(conf_f.read())

    conf =  to_nametuple(conf)
    print(conf)

    print("Loading Vocab", conf.corpus.vocab_path)
    vocab = WordPieceVocab()
    vocab.load(conf.corpus.vocab_path)
    print("Vocab Size: ", len(vocab))

    if conf.context.context_flag:
        print("Loading Context", conf.context.context_path)
        context_vocab = Vocab().load(conf.context.context_path)
        print("Context Size: ", len(context_vocab))
    else:
        context_vocab = None

    if conf.context.context_flag:
        print("Building Context BERT model")
        bert = ContextBERT(len(vocab), len(context_vocab), hidden=conf.model.hidden, n_layers=conf.model.layers, attn_heads=conf.model.attn_heads)
    else:
        print("Building BERT model")
        bert = BERT(len(vocab), hidden=conf.model.hidden, n_layers=conf.model.layers, attn_heads=conf.model.attn_heads)

    print("Creating model Trainer")
    trainer = BERTTrainer(bert, 
                          lr                =conf.train.learning_rate, 
                          betas             =(conf.train.adam_beta1, conf.train.adam_beta2), 
                          weight_decay      =conf.train.adam_weight_decay,
                          with_cuda         =conf.train.with_cuda, 
                          cuda_devices      =conf.train.cuda_devices, 
                          log_freq          =conf.train.log_freq)

    
    for epoch in range(conf.train.epochs):
        print(f"Starting training epoch {epoch}")
        print("Setting up train dataset", conf.corpus.train_dataset)
        train_dataset = MultifileDataset(conf.corpus.train_dataset, 
                                    vocab, 
                                    max_seq_len     =conf.corpus.max_seq_len, 
                                    min_seq_len     =conf.corpus.min_seq_len,
                                    max_tokens_len  =conf.corpus.token_len,
                                    max_word_len    =conf.corpus.max_word_len,
                                    corpus_lines    =conf.corpus.train_corpus_lines,
                                    context         =conf.context.context_flag, 
                                    context_vocab   =context_vocab)

        print("Creating dataloader")
        train_data_loader = DataLoader(train_dataset, 
                                    batch_size   =conf.train.batch_size, 
                                    num_workers  =conf.train.num_workers)


        print("Setting up test dataset", conf.corpus.test_dataset)
        if conf.corpus.test_dataset is not None:
            test_dataset = MultifileDataset(conf.corpus.test_dataset, 
                                            vocab, 
                                            max_seq_len     =conf.corpus.max_seq_len, 
                                            min_seq_len     =conf.corpus.min_seq_len,
                                            max_tokens_len  =conf.corpus.token_len,
                                            max_word_len    =conf.corpus.max_word_len,
                                            corpus_lines    =conf.corpus.test_corpus_lines,
                                            context         =conf.context.context_flag, 
                                            context_vocab   =context_vocab)

            test_data_loader = DataLoader(test_dataset, 
                                        batch_size    =conf.train.batch_size, 
                                        num_workers   =conf.train.num_workers)
        else:
            test_data_loader = None   


        trainer.setup_dataloaders(train_data_loader, test_data_loader)

        trainer.train(epoch, conf.train.log_path)
        trainer.save(epoch, conf.train.output_path)

        if test_data_loader is not None:
            trainer.test(epoch, conf.train.log_path)

if __name__ == '__main__':
    train()
  