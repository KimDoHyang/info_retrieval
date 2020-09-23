import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import math
import operator
import numpy as np


# CLI - USER INPUT VERSION

# TF-IDF 계산식
def tf_idf(tf, N, df):
    return (math.log10(1 + tf)) * (math.log10(N / df))


# Cosine Similarity 계산식
def cosine_similarity(x, y):
    normalizing_factor_x = np.sqrt(np.sum(np.square(x)))
    normalizing_factor_y = np.sqrt(np.sum(np.square(y)))
    return np.matmul(x, np.transpose(y) / (normalizing_factor_x * normalizing_factor_y))


# 리스트 공통 원소 추출 함수
def common_list(x_list, y_list):
    return list(set(x_list) & set(y_list))


# 실행함수
def search_keyword():
    input_data = str(input())
    print(f'\n 검색 중입니다...\n')
    search = DocumentTermMatrix()
    additional_stopwords = ('.', ',', '\'s', '!', ':', '[', ']', '?', ';', '``', 'a', 'an',  '#', '--', "\''", "\'", '(', ')')
    search.add_stopwords(additional_stopwords)
    print(search.rank_documentations(input_data))


# 객체화
class DocumentTermMatrix:
    def __init__(self):
        self.stopword_list = set(stopwords.words('english'))
        self.snowball_stemmer = SnowballStemmer("english")

        # document path
        self.path = os.path.join('./', 'data')
        self.file_list = sorted(os.listdir(path=self.path), key=lambda x: int(x.split('.')[0]))

        # query path
        self.query_path = os.path.join('./', 'query')
        self.query_file = os.listdir(path=self.query_path)

        # save document-term matrix with tf-idf
        self.tf_idf_matrix = []

    # additional stopwords
    def add_stopwords(self, additional_words):
        self.stopword_list.update(additional_words)

    # document path update
    def update_path(self, new_path):
        self.path = new_path
        self.file_list = sorted(os.listdir(path=self.path), key=lambda x: int(x.split('.')[0]))

    # query path update
    def update_query_path(self, new_path):
        self.query_path = new_path
        self.query_file = os.listdir(path=self.query_path)

    # update language for stemmer
    def update_stemmer_language(self, language):
        self.stopword_list = set(stopwords.words(language))
        self.snowball_stemmer = SnowballStemmer(language)

    # index
    def all_terms(self):

        # all terms in docs
        term_list = []

        # 경로 내의 모든 파일에서 사용된 모든 토큰 리스트 생성
        for index in range(0, len(self.file_list)):
            with open(f"{self.path}/{self.file_list[index]}", 'r') as f:
                data = f.read()
            # tokenizing + filter stopwords + stemming
            tokens = [self.snowball_stemmer.stem(token) for token in nltk.word_tokenize(data)
                      if token not in self.stopword_list]
            text = nltk.Text(tokens)
            term_list += list(text.vocab().keys())

        # all terms in docs without duplicate
        terms = sorted(list(set(term_list)))
        return terms

    # index
    def index_dictionary(self):

        # Index Dictionary
        index_dictionary = {}

        # 경로 내의 모든 파일에서 사용된 모든 토큰 리스트 생성
        for index in range(0, len(self.file_list)):
            with open(f"{self.path}/{self.file_list[index]}", 'r') as f:
                data = f.read()
            # tokenizing + filter stopwords + stemming
            tokens = [self.snowball_stemmer.stem(token) for token in nltk.word_tokenize(data)
                      if token not in self.stopword_list]
            text = nltk.Text(tokens)
            index_dictionary[self.file_list[index]] = text.vocab()

        return index_dictionary

    # inverted index
    def inverted_dictionary(self):

        # Inverted Index Dictionary
        inverted_dictionary = {}
        all_terms = self.all_terms()
        index_dictionary = self.index_dictionary()

        # inverted index by term created
        for term in all_terms:
            for doc, freq in index_dictionary.items():
                if term in freq:
                    if term not in inverted_dictionary:
                        inverted_dictionary[term] = [doc]
                    else:
                        inverted_dictionary[term] += [doc]

        return inverted_dictionary

    # Document-Term Incidence Matrix by TF-IDF
    def create_matrix(self):

        all_terms = self.all_terms()
        inverted_dictionary = self.inverted_dictionary()

        # 경로 내의 모든 파일-모든 토큰에 대한 Document-Term Incidence Matrix
        for index in range(0, len(self.file_list)):
            with open(f"{self.path}/{self.file_list[index]}", 'r') as f:
                data = f.read()
            tokens = [self.snowball_stemmer.stem(token) for token in nltk.word_tokenize(data)
                      if token not in self.stopword_list]
            text = nltk.Text(tokens)

            # 4900여개(모든 토큰 수)의 차원을 가진 영행렬
            token_matrix = np.zeros(len(all_terms))

            if index == 0:
                for index, term in enumerate(all_terms):
                    if term in text.vocab():
                        # TF-IDF로 계산한 Weight를 Matrix로 채우기
                        token_matrix[index] = tf_idf(text.vocab()[term], len(self.file_list),
                                                     len(inverted_dictionary[term]))
                doc_term_matrix = token_matrix

            else:
                for index, term in enumerate(all_terms):
                    if term in text.vocab():
                        token_matrix[index] = tf_idf(text.vocab()[term], len(self.file_list),
                                                     len(inverted_dictionary[term]))
                doc_term_matrix = np.vstack([doc_term_matrix, token_matrix])

        # once we build a matrix and use it without additional calculation
        self.tf_idf_matrix = doc_term_matrix
        return doc_term_matrix

    # get matrix we already built before
    def get_doc_term_matrix(self):
        if not self.tf_idf_matrix:
            self.tf_idf_matrix = self.create_matrix()
        return self.tf_idf_matrix

    # Query Doc-Term Incidence Matrix by TF-IDF
    def create_query_matrix(self, input_data):

        all_terms = self.all_terms()
        inverted_dictionary = self.inverted_dictionary()

        # 경로 내의 query파일 토큰에 대한 Document-Term Incidence Matrix
        #         with open(f"{self.query_path}/{self.query_file[0]}", 'r') as f:
        #             data = f.read()
        data = input_data
        tokens = [self.snowball_stemmer.stem(token) for token in nltk.word_tokenize(data)
                  if token not in self.stopword_list]
        text = nltk.Text(tokens)

        # 4900여개(모든 토큰 수)의 차원을 가진 영행렬
        token_matrix = np.zeros(len(all_terms))

        for index, term in enumerate(all_terms):
            if term in text.vocab():
                # TF-IDF로 계산한 Weight를 Matrix로 채우기
                token_matrix[index] = tf_idf(text.vocab()[term], len(self.file_list), len(inverted_dictionary[term]))
        doc_term_matrix = token_matrix

        return doc_term_matrix

    # Rank with Cosine Similarity between query&docs
    def rank_documentations(self, input_data):

        inverted_dictionary = self.inverted_dictionary()
        documentation_matrix = self.get_doc_term_matrix()
        query_matrix = self.create_query_matrix(input_data)

        rank = {}

        # 경로 내의 query파일 읽어들여 검색 키워드 확인하고 tokenize
        #         with open(f"{self.query_path}/{self.query_file[0]}", 'r') as f:
        #             data = f.read()
        data = input_data
        k_tokens = [self.snowball_stemmer.stem(token) for token in nltk.word_tokenize(data)
                    if token not in self.stopword_list]

        # N개의 키워드가 주어질 때, 해당 키워드들을 모두 포함하고 있는 문서 찾기
        commons = []

        for i in range(len(k_tokens)):
            if i == 0:
                commons = inverted_dictionary[k_tokens[i]]
            elif i > 0:
                commons = common_list(commons, inverted_dictionary[k_tokens[i]])

        common_indexes = sorted([int(doc.split('.')[0]) + 1 for doc in commons], key=int)
        for index in common_indexes:
            rank[index - 1] = cosine_similarity(query_matrix, documentation_matrix[index])

        return dict(sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[:5])


if __name__ == '__main__':
    print(f'검색어를 입력하세요\n')
    search_keyword()