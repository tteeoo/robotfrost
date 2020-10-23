from torch import tensor
from torch.utils.data import Dataset
from collections import Counter
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers

class GutenbergDataset(Dataset):
    """A class for easily processing our data."""

    # Dunder methods:
    def __init__(self, seq_len):
        self.seq_len = seq_len
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()

        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def __len__(self):
        return len(self.words_indexes) - self.seq_len

    def __getitem__(self, index):
        return (
            tensor(self.words_indexes[index:index+self.seq_len]),
            tensor(self.words_indexes[index+1:index+self.seq_len+1]),
        )

    def load_words(self):
        """Returns the dataset as an array of words."""

        # The IDs of ebooks from http://www.gutenberg.org/
        id_list = [3021, 3026, 29345, 58611]

        # Get and preprocess the data:
        text = ' '
        for id_nr in id_list:
            text += strip_headers(load_etext(id_nr)).strip()
        for i in '123456890[]()"”“\t\n':
            text = text.replace(i, '')
        text = text.replace('--', '')
        return [i for i in text.split(' ') if i != '']

    def get_uniq_words(self):
        """Returns a sorted array of the unique words in the dataset."""

        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

