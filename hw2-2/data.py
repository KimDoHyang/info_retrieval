import csv
import re

class Data:
    # Load data from the given csv file path
    def __init__(self, path, n_gram):
        input = open(path, 'r')
        csv_reader = csv.reader(input)
        classes = []
        titles = []
        titles_original = []
        descriptions = []
        for line in csv_reader:
            classes.append(int(line[0]) - 1)

            titles.append(
                self.extract_word_grams(self.repair_string(line[1]), n_gram)
            )
            titles_original.append(line[1])

            descriptions.append(
                self.extract_word_grams(self.repair_string(line[2]), n_gram)
            )

        input.close()

        self.classes = classes
        self.titles = titles
        self.titles_original = titles_original
        self.descriptions = descriptions
        self.length = len(classes)

    # Repair string with pure lowercase alphabets.
    def repair_string(self, str):
        # Remove all non-alphabetic characters
        alphabetic = re.sub('[^a-zA-Z]', ' ', str)

        # Lower upper cases
        return alphabetic.lower()

    # Apply word n-grams to string
    def extract_word_grams(self, str, n_gram):
        words = str.split()
        grams = []
        for i in range(len(words) - n_gram + 1):
            grams.append(
                '-'.join(words[i:i+n_gram])
            )

        return grams
