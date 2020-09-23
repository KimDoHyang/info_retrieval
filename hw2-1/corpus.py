from collections import Counter
from random import random, shuffle
from subword import extract_subwords

class Corpus:
    def __init__(self, corpus_path, gram_min, gram_max, use_part=False):
        # Load and corpus
        print("loading...")
        if use_part:
            text = open(corpus_path, mode='r').readlines()[0][:1000000]  # Load a part of corpus for debugging
        else:
            text = open(corpus_path, mode='r').readlines()[0]  # Load full corpus for submission

        print("preprocessing...")
        words_seq = text.split()

        freqs = Counter(words_seq)
        words = set([])

        # Get words set from corpus
        for word, freq in freqs.items():
            # Discard rare words
            if freq > 4:
                words.add(word)

        self.words = words
        self.number_of_words = len(words)
        print(f"# of total words: {self.number_of_words}")

        # Give an index number to a word
        w2i = {}
        i2w = {}
        for (index, word) in enumerate(words):
            w2i[word] = index
            i2w[index] = word

        self.i2w = i2w

        corpus_seq = []
        for word in words_seq:
            if word in w2i:
                corpus_seq.append(w2i[word])

        self.corpus_word_index_seq = corpus_seq
        self.corpus_length = len(corpus_seq)
        print(f"length of corpus: {len(corpus_seq)}")
        self.word_freqs = Counter(corpus_seq)

        # Extract subwords from each word
        subwords = {}
        for word in words:
            subwords[word] = extract_subwords(word, gram_min, gram_max)

        # Get hash values from subwords
        sw_hashes = {}
        total_hashes = set([])
        sw2h = {}
        for (word, subwords) in subwords.items():
            word_index = w2i[word]
            sw_hashes[word_index] = []
            for subword in subwords:
                hash_val = fnv_hash(subword)
                sw_hashes[word_index].append(hash_val)
                total_hashes.add(hash_val)
                sw2h[subword] = hash_val

        # Convert from hash values to indices
        h2i = {}
        for (index, hash_val) in enumerate(total_hashes):
            h2i[hash_val] = index

        sw2i = {}
        for (subword, hash_val) in sw2h.items():
           sw2i[subword] = h2i[hash_val]

        self.sw2i = sw2i

        self.number_of_subwords = len(total_hashes)
        print(f"# of total subwords: {self.number_of_subwords}")

        sw_indices = {}
        for (word_index, hashes) in sw_hashes.items():
            sw_indices[word_index] = []
            for hash_val in hashes:
                sw_indices[word_index].append(h2i[hash_val])

        self.subword_indices = sw_indices


    def word_of_index(self, index):
        return self.i2w[index]


    def get_subword_indices(self, word_index):
        if word_index in self.subword_indices:
            return self.subword_indices[word_index]
        else:
            return []


    def word_frequency(self, word_index):
        if word_index in self.word_freqs:
            return self.word_freqs[word_index]
        else:
            return 0


    def index_of_subword(self, subword):
        if subword in self.sw2i:
            return self.sw2i[subword]
        else:
            return None


    # Return Skip-gram training set (input_seq, output_seq)
    def generate_training_data(self, window_size=5, subsampling=True):
        subsample_threshold = 0.001
        input_seq = []
        output_seq = []

        for (center_index, word) in enumerate(self.corpus_word_index_seq):
            if subsampling:
                f = self.word_frequency(word) / self.corpus_length
                prob = (subsample_threshold / f) ** 0.5
                if not gacha(prob):
                    continue

            subword_indices = self.get_subword_indices(word)

            leftmost_index = max(center_index - window_size, 0)
            for context_index in range(leftmost_index, center_index):
                input_seq.append(subword_indices)
                output_seq.append(self.corpus_word_index_seq[context_index])

            rightmost_index = min(len(self.corpus_word_index_seq), center_index + window_size + 1)
            for context_index in range(center_index + 1, rightmost_index):
                input_seq.append(subword_indices)
                output_seq.append(self.corpus_word_index_seq[context_index])

        return input_seq, output_seq


def fnv_hash(str, k=2**21):
    hval = 0x811c9dc5
    fnv_32_prime = 0x01000193
    for s in str:
        hval = hval ^ ord(s)
        hval = (hval * fnv_32_prime) % k
    return hval


def gacha(prob):
    if random() <= prob:
        return True
    else:
        return False
