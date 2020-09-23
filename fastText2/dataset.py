import csv
import re


class DataSet:
    # Load data from the given csv file path
    def __init__(self, path, n_gram):
        input = open(path, 'r')
        data = csv.reader(input)
        class_list = []
        title_list = []
        title_original_list = []
        description_list = []
        for line in data:
            class_list.append(int(line[0]) - 1)
            title_list.append(self.get_word_grams(line[1], n_gram))
            title_original_list.append(line[1])
            description_list.append(self.get_word_grams(line[2], n_gram))

        input.close()

        self.classes = class_list
        self.titles = title_list
        self.titles_original = title_original_list
        self.descriptions = description_list
        self.length = len(class_list)

    def get_word_grams(self, corpus, n_gram):
        # 특수문자 제외, 모두 소문자로
        corpus_extracted = re.sub('[^a-zA-Z]', ' ', corpus)
        lower_corpus = corpus_extracted.lower()

        # 워드 그램 추출
        words = lower_corpus.split()
        grams = [
            '-'.join(words[i:i + n_gram])
            for i in range(len(words) - n_gram + 1)
        ]

        return grams


class ClassLabel:
    # Load classes
    def __init__(self, path):
        file = open(path, 'r')
        class_names = file.read().split()
        file.close()

        i2c = {}
        for index, name in enumerate(class_names):
            i2c[index] = name

        self.class_names = i2c
        self.number_of_classes = len(class_names)

    def class_name(self, index):
        if index in self.class_names:
            return self.class_names[index]
        else:
            return None
