import argparse
from corpus import Corpus
from train import train_fasttext
from experiment import find_similar_words

def get_args():
    parser = argparse.ArgumentParser(description='FastText')
    parser.add_argument('ns', metavar='negative-samples', type=int,
                        help='Number of negative samples')
    parser.add_argument('gram_min', metavar='gram-min', type=int,
                        help='Min length of subwords')
    parser.add_argument('gram_max', metavar='gram-max', type=int,
                        help='Max length of subwords')
    parser.add_argument('part', metavar='partition', type=str,
                        help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')

    return parser.parse_args()


def main():
    args = get_args()
    corpus = Corpus("text8", args.gram_min, args.gram_max, args.part=="part")
    subword_embeddings, _ = train_fasttext(corpus, ns_num=args.ns, window_size=5, dimension=100, learning_rate=0.01, epoch=1, subsampling=True)
    find_similar_words(corpus, subword_embeddings, args.gram_min, args.gram_max)


main()
