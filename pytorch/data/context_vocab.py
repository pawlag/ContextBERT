import pickle
from collections import Counter


class BaseVocab(object):
    """Defines a vocabulary object that will be used to numericalize a field.
    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
    """

    def __init__(self, counter, max_size=None, min_freq=1, specials=['[PAD]', '[UNK]']):
        """Create a Vocab object from a collections.Counter.
        Arguments:
            counter: collections.Counter object holding the frequencies of
                each value found in the data.
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary in addition to an <unk>
                token. Default: ['[PAD]', [UNK]]
        """
        self.freqs = counter
        self.max_size = max_size
        self.min_freq = min_freq
        self.specials = specials
        self.stoi = {}
        self.itos = []

    def build(self):
        """
        Create a Vocab object from counter.
        """
        counter = self.freqs.copy()
        min_freq = max(self.min_freq, 1)

        self.itos = list(self.specials)
        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in self.specials:
            del counter[tok]

        max_size = None if self.max_size is None else self.max_size + len(self.itos)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}


    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def vocab_rerank(self):
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1


class Vocab(BaseVocab):
    def __init__(self, counter=Counter(), max_size=None, min_freq=1):
        self.pad_index = 0
        self.unk_index = 1
        super().__init__(counter,  max_size=max_size, min_freq=min_freq)

    @staticmethod
    def load(vocab_path: str) -> 'Vocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)



