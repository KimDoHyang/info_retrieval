import argparse
import copy
from random import random
from collections import Counter
import torch


def get_args():
    parser = argparse.ArgumentParser(description='FastText')
    parser.add_argument('ns', metavar='negative-samples', type=int,
                        help='Number of negative samples')
    parser.add_argument('gram_min', metavar='gram-min', type=int,
                        help='Min length of subwords')
    parser.add_argument('gram_max', metavar='gram-max', type=int,
                        help='Max length of subwords')

    return parser.parse_args()


def main():
    args = get_args()

    # get text file
    text = open('text8', mode='r').readlines()[0][:1000000]

    # raw word list of given text file
    words_sequence = text.split()
    frequency = Counter(words_sequence)

    # filtered word list of given text file
    # not duplicated and filtered(at least 5 times)
    word_list = set([])

    # Get wordset filtered by frequency(4 times)
    for word, freq in frequency.items():
        # discard rare words
        if freq > 4:
            word_list.add(word)

    num_of_words = len(word_list)

    word2index = {}
    index2word = {}
    for (index, word) in enumerate(word_list):
        word2index[word] = index
        index2word[index] = word

    # index of each word in all text words
    word_index_sequence = []
    for word in words_sequence:
        if word in word2index:
            word_index_sequence.append(word2index[word])

    word_index_sequence_length = len(word_index_sequence)
    word_index_frequency = Counter(word_index_sequence)

    # subword_extract function definition
    def subword_extract(word, gram_min, gram_max):
        subwords_list = []
        # word wrapped with token <>
        wrapped_word = f'<{word}>'
        wrapped_length = len(wrapped_word)
        for n in range(gram_min, min(gram_max+1, wrapped_length+1)):
            zip_word_list = [wrapped_word[i:] for i in range(n)]
            subwords = list(zip(*zip_word_list))
            subwords_list += list(map("".join, subwords))
        if wrapped_word not in subwords_list:
            subwords_list.append(wrapped_word)
        return subwords_list

    word2subword = {}
    gram_min = args.gram_min
    gram_max = args.gram_max
    for word in word_list:
        word2subword[word] = subword_extract(word, gram_min, gram_max)

    # all the subword hash values set(not duplicated)
    total_hashes = set([])
    # word_index to subword_hash value list
    word_index2subword_hashes = {}

    # subword to hash value
    subword2hash = {}

    # FNV-1a Hash function
    def fnv32a(str):
        hash_value = 0x811c9dc5
        fnv_32_prime = 0x01000193
        unit32_max = 2 ** 32
        for s in str:
            # ord() 문자를 ASCII code로 변환
            hash_value = hash_value ^ ord(s)
            hash_value = (hash_value * fnv_32_prime) % unit32_max
        return hash_value

    # Get hash values with FNV-1a hash functions from subwords
    for word, subwords in word2subword.items():
        word_index2subword_hashes[word2index[word]] = []
        for subword in subwords:
            hash_value = fnv32a(subword)
            total_hashes.add(hash_value)
            word_index2subword_hashes[word2index[word]].append(hash_value)
            subword2hash[subword] = hash_value

    num_of_subwords = len(total_hashes)

    hash2subword_index = {}
    for (index, hash_value) in enumerate(total_hashes):
        hash2subword_index[hash_value] = index

    subword2subword_index = {}
    for (subword, hash_value) in subword2hash.items():
        subword2subword_index[subword] = hash2subword_index[hash_value]

    # convert from subword hash values to hash_indices
    subword_hash2hash_index = {}
    for index, hash_value in enumerate(total_hashes):
        subword_hash2hash_index[hash_value] = index

    # deep copy of word_index to subword_hash value list
    # the copied will be used to store replaced values(hash value list > hash value index list)
    word_index2subword_hash_indices = copy.deepcopy(word_index2subword_hashes)

    for word_index in list(word_index2subword_hash_indices.keys()):
        word_index2subword_hash_indices[word_index] = [
            subword_hash2hash_index[hash_value] for hash_value in word_index2subword_hash_indices[word_index]
        ]

    # generating training data
    def generate_training_data(window_size=5, subsampling=True):
        subsampling_threshold = 0.001
        input_seq = []
        output_seq = []

        for center_index, word_index in enumerate(word_index_sequence):
            if subsampling:
                if word_index in word_index_frequency:
                    f = word_index_frequency[word_index] / word_index_sequence_length
                else:
                    f = 0
                prob = (subsampling_threshold / f) ** 0.5
                if not random() <= prob:
                    continue

            subword_indices = word_index2subword_hash_indices[word_index]
            left_index = max(center_index - window_size, 0)
            # 어떤 Word의 앞-뒤로 어떤 단어가 나오는지, 이를 subword로 학습할 것이므로
            # 각 word가 output일때, input은 centerword의 subword list(hash value index set)이 된다.
            for context_index in range(left_index, center_index):
                # input > 해당 word의 subword들의 hash value값 index들
                input_seq.append(subword_indices)
                # 해당 word의 index
                output_seq.append(word_index_sequence[context_index])
            right_index = min(word_index_sequence_length, center_index + window_size + 1)
            for context_index in range(center_index + 1, right_index):
                input_seq.append(subword_indices)
                output_seq.append(word_index_sequence[context_index])

        return input_seq, output_seq

########################################################################

    # Negative Sampling Class: Randomly choose negative samples according to given prob weights
    # random.choices(with weights)가 가장 간단한 방법이지만 해당 방법은 비효율적이고 느림
    # Binary Search를 사용해 sampling할 것
    class NegativeSampling:
        def __init__(self, prob_weights):
            total_weight = sum(prob_weights)
            bounds = []
            acc = 0
            for (index, weight) in enumerate(prob_weights):
                bounds.append(acc)
                acc += weight / total_weight
            bounds.append(1.0)

            self.number_of_candidates = len(prob_weights)
            self.bounds = bounds

        def get_negative_samples(self, number_of_ns, positive_index):
            count = 0
            negative_samples = []
            while count < number_of_ns:
                new_sample = self.choose_one()
                if new_sample == positive_index:
                    continue
                negative_samples.append(new_sample)
                count += 1

            return negative_samples

        # Randomly choose one sample according to prob weights
        def choose_one(self):
            value = random()
            leftmost = 0
            rightmost = self.number_of_candidates - 1

            # Binary search to find upper bound about the random value
            while leftmost <= rightmost:
                mid = (leftmost + rightmost) // 2
                if self.bounds[mid] > value:
                    rightmost = mid - 1
                elif self.bounds[mid] <= value:
                    upper = mid
                    leftmost = mid + 1
            return upper

#######################################################################

    # skipgram function for negative sampling
    def skipgram_ns(subwords, input_matrix, output_matrix):
        # subwords의 vector를 1차원 벡터로 합쳐주는 과정 (S,D) > (D)
        hidden_layer = torch.sum(input_matrix[subwords], 0) # torch.tensor(D)


        scores = torch.squeeze(
            torch.mm(output_matrix, torch.unsqueeze(hidden_layer, 1))
        ) # torch.tensor(K)

        # negative loglikelihood이므로 음수를 취해준다
        loss_scores = -scores
        # 해당 score의 0번째는 positive index의 score에 해당한다.
        loss_scores[0] *= -1
        log_sigmoids = torch.log(torch.sigmoid(loss_scores))

        loss = -torch.sum(log_sigmoids)

        grad_vh = torch.sigmoid(scores) # torch.tensor(K)
        grad_vh[0] -= 1

        grad_out = torch.mm(
            grad_vh.view(-1, 1), # torch.tensor(K,1)
            hidden_layer.view(1, -1) # torch.tensor(1,D)
        ) # torch.tensor(K,D)

        grad_in = torch.mm(
            grad_vh.view(1, -1), # torch.tensor(1,K)
            output_matrix #torch.tensor(K,D)
        ) # torch.tensor(1,D)

        return loss, grad_in, grad_out

    # Train function
    def train(ns=10, window_size=5, dimension=100, learning_rate=0.02, epoch=3, subsampling=True):
        # Xavier initialization of weight matrix
        # W_in : subword to word
        W_in = torch.randn(num_of_subwords, dimension) / (dimension ** 0.5)
        # W_out : word to ~
        W_out = torch.randn(num_of_words, dimension) / (dimension ** 0.5)

        count = 0
        losses = []
        # frequency에 0.75(3/4)을 제곱해주면 negative sampling 효과가 개선된다
        prob_weight = [
            (word_index_frequency[word_index] / word_index_sequence_length) ** 0.75
            for word_index in list(word_index_frequency.keys())
        ]

        sampler = NegativeSampling(prob_weight)

        for epoch_index in range(epoch):
            print(f'Epoch {epoch_index + 1}')
            input_seq, output_seq = generate_training_data(window_size, subsampling)
            print(f'The number of training data : {len(input_seq)}')

            print('Loss : ')
            for inputs, output in zip(input_seq, output_seq):
                count += 1
                # 학습될 output 선택 with negative sampling
                # 학습 대상인 해당 word + negative sample 결과만을 학습
                activated = [output] + sampler.get_negative_samples(ns, output)
                Loss, G_in, G_out = skipgram_ns(inputs, W_in, W_out[activated])
                # Gradient descent : Weight Vector - learning rate * 기울기
                W_in[inputs] -= learning_rate * G_in.squeeze()
                W_out[activated] -= learning_rate * G_out

                losses.append(Loss.item())

                if count % 50000 == 0:
                    avg_loss = sum(losses) / len(losses)
                    print("%f" % (avg_loss), end=", ", flush=True)
                    losses = []

        print("training finished!")
        return W_in, W_out

######################################################################
    # Test
    testwords = [
        'narrow-mindedness',
        'department',
        'campfires',
        'knowing',
        'urbanize',
        'imperfection',
        'principality',
        'abnormal',
        'secondary',
        'ungraceful'
    ]

    def get_word_embeddings(subword_embeddings):
        D = subword_embeddings.size(1)
        word_embeddings = torch.zeros(num_of_words, D)

        for word_index in range(num_of_words):
            subword_indices = word_index2subword_hash_indices[word_index]
            word_embeddings[word_index] = torch.sum(subword_embeddings[subword_indices], 0)

        return word_embeddings # torch.tensor(V, D)

    def find_similar_words(subword_embeddings, gram_min, gram_max):
        word_embeddings = get_word_embeddings(subword_embeddings)

        def get_testword_embedding(testword):
            subwords = subword_extract(testword, gram_min, gram_max)

            subword_indices = []
            for subword in subwords:
                subword_index = subword2subword_index[subword]
                if subword_index:
                    subword_indices.append(subword_index)

            return torch.sum(subword_embeddings[subword_indices], 0) # torch.tensor(D)

        for testword in testwords:
            testword_emb = get_testword_embedding(testword)

            similarities = [(torch.cosine_similarity(testword_emb, word_emb, 0), index) for index, word_emb in enumerate(word_embeddings)]

            similarities.sort(key=lambda elem: elem[0], reverse=True)
            print()
            print("-----------------------------------------------")
            print("The most similar words of \"" + testword + "\"")
            count = 0
            i = 0
            while count < 5:
                similarity, word_index = similarities[i]
                i += 1
                if index2word[word_index] == testword:
                    continue
                print(index2word[word_index] + ":%.3f" % (similarity))
                count += 1
            print("-----------------------------------------------")
            print()

#####################################################################
    subword_embeddings, _ = train(ns=args.ns, window_size=5, dimension=100, learning_rate=0.02, epoch=1, subsampling=True)
    find_similar_words(subword_embeddings, args.gram_min, args.gram_max)
#####################################################################


main()
