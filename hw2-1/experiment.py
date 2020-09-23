import torch
from datetime import datetime
from subword import extract_subwords

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


def get_word_embeddings(corpus, subword_embeddings):
    D = subword_embeddings.size(1)
    word_embeddings = torch.zeros(corpus.number_of_words, D)

    for word_index in range(corpus.number_of_words):
        subword_indices = corpus.get_subword_indices(word_index)
        word_embeddings[word_index] = torch.sum(subword_embeddings[subword_indices], 0)

    return word_embeddings # torch.tensor(V, D)


def save_subword_indices(corpus, current_date):
    output = open(f"embeddings/subword-indices-{current_date}", 'w')
    for (subword, index) in corpus.sw2i.items():
        output.write(f"{subword} {index}\n")

    output.close()


def save_word_indices(corpus, current_date):
    output = open(f"embeddings/word-indices-{current_date}", 'w')
    for (index, word) in corpus.i2w.items():
        output.write(f"{word} {index}\n")

    output.close()


def find_similar_words(corpus, subword_embeddings, gram_min, gram_max):
    # Save subword embedding matrix
    current_date = datetime.now()
    torch.save(subword_embeddings, f"embeddings/subword-embeddings-{current_date}")
    save_subword_indices(corpus, current_date)

    # Save word embedding matrix
    word_embeddings = get_word_embeddings(corpus, subword_embeddings)
    torch.save(word_embeddings, f"embeddings/word-embeddings-{current_date}")
    save_word_indices(corpus, current_date)

    def get_testword_embedding(testword):
        subwords = extract_subwords(testword, gram_min, gram_max)

        subword_indices = []
        for subword in subwords:
            subword_index = corpus.index_of_subword(subword)
            if subword_index:
                subword_indices.append(subword_index)

        return torch.sum(subword_embeddings[subword_indices], 0) # torch.tensor(D)


    for testword in testwords:
        testword_emb = get_testword_embedding(testword)

        similarities = [(torch.cosine_similarity(testword_emb, word_emb, 0), index) for index, word_emb in enumerate(word_embeddings)]

        similarities.sort(key=lambda elem: elem[0], reverse=True)
        print()
        print("===============================================")
        print("The most similar words to \"" + testword + "\"")
        count = 0
        i = 0
        while count < 5:
            similarity, word_index = similarities[i]
            i += 1
            if corpus.word_of_index(word_index) == testword:
                continue
            print(corpus.word_of_index(word_index) + ":%.3f" % (similarity))
            count += 1
        print("===============================================")
        print()

