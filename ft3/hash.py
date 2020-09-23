def generate_gram2index(data):
    # Make word n-grams to hash dictionary
    g2h = {}
    for title in data.titles:
        for gram in title:
            g2h[gram] = fnv_hash(gram)

    for desc in data.descriptions:
        for gram in desc:
            g2h[gram] = fnv_hash(gram)

    # Label each hash value with unique index
    hashes = set([])
    for _, hash_val in g2h.items():
        hashes.add(hash_val)

    h2i = {}
    for index, hash_val in enumerate(hashes):
        h2i[hash_val] = index

    # Make word n-grams to index dictionary
    g2i = {}
    for gram, hash_val in g2h.items():
        g2i[gram] = h2i[hash_val]

    return g2i, len(hashes)

def fnv_hash(str, k=2**21):
    hval = 0x811c9dc5
    fnv_32_prime = 0x01000193
    for s in str:
        hval = hval ^ ord(s)
        hval = (hval * fnv_32_prime) % k

    return hval
