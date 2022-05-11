import yaml
import argparse

from torch.utils.data import DataLoader
from model.bert import BERT
from model.contextual_bert import ContextBERT
from model.trainer import BERTTrainer
from data import MultifileDataset, WordPieceVocab, ContextVocab


def train():

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str, help="path to config file")
    args = parser.parse_args()
    with open(args.config, "r") as conf_f:
        conf = yaml.safe_load(conf_f.read())
    print(conf)

    # parser.add_argument("-trt", "--train_dataset", required=True, type=str, help="train dataset for train bert")
    # parser.add_argument("-tst", "--test_dataset", type=str, default=None, help="test set for evaluate train set")
    # parser.add_argument("-vp", "--vocab_path", required=True, type=str, help="path of built vocab model for corpus")
    # parser.add_argument("--max_seq_len", type=int, default=20, help="maximum sequence len")
    # parser.add_argument("--min_seq_len", type=int, default=3, help="minimum sequence len")
    # parser.add_argument("--token_len", type=int, default=25, help="maximum tokens len")
    # parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    
    # parser.add_argument('--context', action=argparse.BooleanOptionalAction)
    # parser.add_argument("-cp", "--context_path", type=str, help="path of built vocab model for context")

    # parser.add_argument("-hs", "--hidden", type=int, default=256, help="hidden size of transformer model")
    # parser.add_argument("-cs", "--context_embedding_size", type=int, default=128, help="size of context embedding")
    # parser.add_argument("-l", "--layers", type=int, default=8, help="number of layers")
    # parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")

    # parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    # parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    # parser.add_argument("-w", "--num_workers", type=int, default=1, help="dataloader worker size")

    # parser.add_argument("--with_cuda", action=argparse.BooleanOptionalAction, help="training with CUDA ")
    # parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")    

    # parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    # parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    # parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    # parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")

    # parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter default is 10")    
    # parser.add_argument("-o", "--output_path", required=True, type=str, help="output/bert.model")
    

    print("Loading Vocab", args.vocab_path)
    vocab = WordPieceVocab.load_vocab(args.vocab_path)
    print("Vocab Size: ", len(vocab))

    if args.context:
        print("Loading Context", args.context_path)
        context_dict = ContextVocab.load_vocab(args.context_path)
        print("Context Size: ", len(context_dict))
    else:
        context_dict = None

    print("Loading Train Dataset", args.train_dataset)
    train_dataset = MultifileDataset(args.train_dataset, 
                                    vocab, 
                                    max_seq_len     =args.max_seq_len, 
                                    min_seq_len     =args.min_seq_len,
                                    max_tokens_len  =args.token_len,
                                    max_word_len    =   50,
                                    corpus_lines    =args.corpus_line,
                                    context         =args.context, 
                                    context_vocab   = context_dict)

    print("Loading Test Dataset", args.test_dataset)
    test_dataset = MultifileDataset(args.test_dataset, 
                                    vocab, 
                                    max_seq_len     =args.max_seq_len, 
                                    min_seq_len     =args.min_seq_len,
                                    max_tokens_len  =args.token_len,
                                    corpus_lines    =args.corpus_line,
                                    context         =args.context, 
                                    context_vocab   = context_dict)


    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) if test_dataset is not None else None

    if args.context:
        print("Building Context BERT model")
        bert = ContextBERT(len(vocab), len(context_dict), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)
    else:
        print("Building BERT model")
        bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)

    print("Creating model Trainer")
    trainer = BERTTrainer(bert, train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                            lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                            with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq)

    print("Training Start")
    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)

        if test_data_loader is not None:
            trainer.test(epoch)

if __name__ == '__main__':
    train()

    # python train.py -c /mnt/txt_base/files/ipa140710.json -v /home/luser/data/patent/model/vocab.pkl -o /home/luser/data/patent/model/bert.model
    # python train.py -c /mnt/txt_base/files/ipa140710.json -v /home/luser/data/patent/model/vocab.pkl -o /home/luser/data/patent/model/bert.model --context -cp /home/luser/data/patent/model/context.pkl


    