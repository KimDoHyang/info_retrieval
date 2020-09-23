
def extract_subwords(word, gram_min, gram_max):
    wrapped = '<' + word + '>'
    full_len = len(wrapped)

    subwords = []
    subwords.append(wrapped)
    for subword_len in range(gram_min, min(gram_max + 1, full_len)):
        for start_index in range(full_len - subword_len + 1):
            subwords.append(wrapped[start_index : start_index + subword_len])

    return subwords
