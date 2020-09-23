import argparse
from dataset import DataSet, ClassLabel
from train import *
import test


def get_args():
    parser = argparse.ArgumentParser(description='Text Classification')
    parser.add_argument('n_gram', metavar='n-gram', type=int,
                        help='length of word n-gram')
    parser.add_argument('classes_path', metavar='classes-path', type=str,
                        help='file path of class list')
    parser.add_argument('train_path', metavar='train-path', type=str,
                        help='file path of training data (csv)')
    parser.add_argument('test_path', metavar='test-path', type=str,
                        help='file path of test data (csv)')

    return parser.parse_args()


def main():
    args = get_args()
    data = DataSet(args.train_path, n_gram=args.n_gram)
    classes = ClassLabel(args.classes_path)
    training_data = TrainingData(data)
    model = train_classification_model(training_data, classes.number_of_classes, dimension=100, learning_rate=0.02, epoch=10)
    test.test_model(classes, training_data.g2i, model, args.test_path, n_gram=args.n_gram)


main()
