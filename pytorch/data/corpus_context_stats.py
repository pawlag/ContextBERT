import pickle
from collections import Counter
from patent.patent_loader import measure_files
from vocab_wp_hybrid import WordPieceVocab
from context_vocab import Vocab

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
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list(specials)
        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in specials:
            del counter[tok]

        max_size = None if max_size is None else max_size + len(self.itos)

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
    def __init__(self, counter, max_size=None, min_freq=1):
        self.pad_index = 0
        self.unk_index = 1
        super().__init__(counter, specials=["[PAD]", "[UNK]"], max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentece, seq_len, with_eos=False, with_sos=False) -> list:
        pass

    def from_seq(self, seq, join=False, with_pad=False):
        pass

    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)



if __name__ == '__main__':
    
    # get tokenizer
    vb = WordPieceVocab()
    vb.load('/home/luser/data/patent/model/wp_vocab_hybrid_freq10k_75k.json')

    # measure files
    files_path = '/mnt/corpus_files_sample/'    
    context_counter, line_counter = measure_files(files_path, 3, 20, 50)
    print(f"context_counter {context_counter}")
    print(f"line_counter {line_counter}")   

    #context_vb = Vocab(context_counter)
    #context_vb.save_vocab('/home/luser/data/patent/model/context_vb.pck')


    # wynik z 09.05 l
    # line counter 5362975950 (min 3 max 20)
    # contex counter 176

    #Counter({'G06': 747880353, 'A61': 672918311, 'H04': 582941805, 'H01': 354397337, 'C07': 349543238, 'G01': 278307214, 'C12': 273588406, 'G02': 106475863, 'B60': 84551885, 'C08': 82163167, 'H02': 75377957, 'A01': 73255600, 'G03': 71678652, 'B01': 70449944, 'G11': 65666501, 'G09': 64710821, 'F16': 62404764, 'G05': 61308048, 'H05': 60404137, 'B65': 53602752, 'H03': 52744054, 'C09': 51338930, 'B32': 50090618, 'A63': 46864112, 'B29': 39904541, 'B41': 39631916, 'G08': 37094398, 'A47': 33036631, 'B23': 31922774, 'G10': 31077562, 'B05': 30439163, 'F02': 28541261, 'E21': 28018520, 'A23': 25990892, 'G07': 25470392, 'F21': 25465104, 'F01': 23420567, 'C23': 22618486, 'G16': 22513019, 'B62': 20320660, 'B25': 18113061, 'B64': 17959130, 'E04': 17017436, 'C01': 15956602, 'F04': 14682664, 'F25': 14561356, 'C10': 14132372, 'F24': 13096456, 'C11': 12735469, 'C40': 11967368, 'C02': 11414276, 'C22': 10451273, 'E05': 9771618, 'C25': 9442029, 'B22': 8908725, 'C03': 8803077, 'F28': 8584731, 'B21': 8327905, 'B82': 8107018, 'A45': 8040365, 'C04': 7977111, 'D06': 7822572, 'B33': 7762732, 'B08': 7419623, 'F03': 7137804, 'F41': 7018770, 'B24': 7004750, 'A41': 6789621, 'G21': 6704912, 'E02': 6600511, 'B63': 6469817, 'A43': 6300880, 'A24': 6036517, 'G0': 5902161, 'B26': 5854094, 'B66': 5773325, 'E06': 5683098, 'D21': 5442478, 'A62': 5189591, 'B67': 5115505, 'F23': 5048404, 'D04': 5017293, 'Y10': 4512754, 'E01': 4314043, 'H0': 4241550, 'C21': 4149755, 'B28': 3919630, 'D01': 3908041, 'F15': 3902004, 'A6': 3839316, 'C30': 3783642, 'E03': 3753158, 'B61': 3622307, 'C 0': 3535104, 'F05': 3472142, 'G04': 3396868, 'B42': 2917153, 'B31': 2895627, 'A44': 2815937, 'F17': 2717208, 'B81': 2684544, 'B44': 2640174, 'C0': 2517775, 'A21': 2485186, 'B02': 2480636, 'A46': 2431474, 'F42': 2331640, 'B27': 2320627, 'B03': 2268511, 'F26': 2184212, 'B07': 2083611, 'A42': 2012777, 'D02': 1947315, 'C05': 1907198, 'D03': 1849749, 'F27': 1796668, 'C1': 1770080, 'Y02': 1547579, 'B06': 1360976, 'B04': 1296841, 'D05': 1211402, 'B43': 1108605, 'B09': 1017199, 'B30': 1003985, 'A22': 976356, 'F22': 881885, 'C13': 846109, 'B6': 683884, 'G1': 635474, 'D10': 556140, 'B0': 519450, 'C06': 503983, 'B2': 476823, 'B3': 411037, 'B4': 375501, 'A0': 346437, 'B68': 344339, 'F1': 309283, 'D07': 287020, 'F0': 271399, 'A4': 270581, 'F2': 264157, 'C14': 226305, 'E0': 220176, 'C2': 217901, 'G12': 206285, 'C ': 201019, 'E2': 147831, 'Y04': 126662, 'A2': 115416, 'D0': 99582, 'G2': 52099, 'C 1': 39192, 'C3': 36831, 'F4': 34077, 'D2': 29328, 'F 2': 11157, 'H99': 5798, 'C99': 5357, 'B 6': 1865, 'E99': 1531, 'F99': 1414, 'G99': 1402, 'A 6': 912, 'B 2': 801, 'F ': 603, 'A 4': 441, 'B 0': 429, 'B ': 335, 'B99': 334, 'B8': 288, 'H 0': 232, 'C 3': 211, 'F 1': 175, 'D99': 164, 'G 0': 81})