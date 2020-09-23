import torch
from random import shuffle, random
from collections import Counter
import argparse
from huffman import HuffmanCoding
from math import sqrt


class Node:
    def __init__(self, index):
        self.index = index
        self.left = None
        self.right = None


class BuildTree:
    def __init__(self, codes):
        self.root = Node(0)
        self.num_of_nodes = 1
        self.build(codes)

    def build(self, codes):
        for code in codes:
            current_node = self.root
            for bit in code:
                if bit == '0':
                    if current_node.left is None:
                        current_node.left = Node(self.num_of_nodes)
                        self.num_of_nodes += 1

                    current_node = current_node.left

                elif bit == '1':
                    if current_node.right is None:
                        current_node.right = Node(self.num_of_nodes)
                        self.num_of_nodes += 1

                    current_node = current_node.right


def Analogical_Reasoning_Task(embedding, output_name):
    questions = open('questions-words.txt', mode='r').readlines()[1:]

    output = open(output_name, 'w')

    def get_closest_word(vector):
        max = -2
        closest_word = None
        for (word, emb) in embedding.items():
            similarity = torch.cosine_similarity(emb, vector, 0)
            if max < similarity:
                max = similarity
                closest_word = word

        return closest_word

    correct_count = 0
    total_count = 0

    for question in questions:
        list = question.split()
        if list[0]==":":
            continue
            
        total_count += 1
        if total_count % 1000 == 0:
            print(f"question count: {total_count}...")
        for word in list:
            if word.lower() not in embedding.keys():
                output.write(f"{total_count}. No word [{word}] in corpus.\n")
                break
        else:
            vec = embedding[list[1].lower()] - embedding[list[0].lower()] + embedding[list[2].lower()]
            guess = get_closest_word(vec)
            output.write(f"{total_count}. answer: {list[3].lower()}, guess: {guess}\n")
            if guess.lower() == list[3].lower():
                correct_count += 1

    print(f"accuracy: {correct_count / total_count}")
    print(f"{correct_count} correct answers from {total_count} questions.")
    output.write(f"accuracy: {correct_count / total_count}\n{correct_count} correct answers from {total_count} questions.\n")
    output.close()


def subsampling(word_seq, threshold=0.001):
    stats = Counter(word_seq)
    words = set(word_seq)

    seq_len = len(word_seq)
    sample = []
    for word in words:
        frequency = stats[word] / seq_len

        if random() <= sqrt(threshold / frequency):
            sample.append(word)

    subsample = set(sample)

    return subsample


def skipgram_HS(centerWord, contextCode, inputMatrix, outputMatrix):
    projection = inputMatrix[centerWord]

    products = torch.mm(outputMatrix, torch.unsqueeze(projection, 1))

    loss = 0
    grad_out = torch.zeros(outputMatrix.shape)
    grad_h = torch.zeros(inputMatrix.shape[1])
    for (index, bit) in enumerate(contextCode):
        sigmoid_val = torch.sigmoid(products[index])
        if bit == '0':
            loss -= torch.log(sigmoid_val)
            grad_out[index] = (sigmoid_val - 1) * projection
            grad_h += (sigmoid_val - 1) * outputMatrix[index]

        elif bit == '1':
            loss -= torch.log(1 - sigmoid_val)
            grad_out[index] = sigmoid_val * projection
            grad_h += sigmoid_val * outputMatrix[index]

    grad_in = torch.unsqueeze(grad_h, 0)

    return loss, grad_in, grad_out


def skipgram_NS(centerWord, inputMatrix, outputMatrix):
    K = outputMatrix.size(0)

    projection = inputMatrix[centerWord]
    scores = torch.squeeze(
        torch.mm(outputMatrix, torch.unsqueeze(projection, 1))
    )

    loss_scores = -scores
    loss_scores[0] *= -1
    log_sigmoids = torch.log(torch.sigmoid(loss_scores))

    loss = -torch.sum(log_sigmoids)

    grad_vh = torch.sigmoid(scores)
    grad_vh[0] -= 1

    grad_out = torch.mm(
        grad_vh.view(-1, 1),
        projection.view(1, -1)
    )

    grad_in = torch.mm(
        grad_vh.view(1, -1),
        outputMatrix
    )

    return loss, grad_in, grad_out


def CBOW_HS(contextWords, centerCode, inputMatrix, outputMatrix):
    D = inputMatrix.size(1)

    projection = torch.zeros(D)
    for word in contextWords:
        projection += inputMatrix[word]

    products = torch.mm(outputMatrix, torch.unsqueeze(projection, 1))

    loss = 0
    grad_out = torch.zeros(outputMatrix.shape)
    grad_h = torch.zeros(D)
    for (index, bit) in enumerate(centerCode):
        sigmoid_val = torch.sigmoid(products[index])
        if bit == '0':
            loss -= torch.log(sigmoid_val)
            grad_out[index] = (sigmoid_val - 1) * projection
            grad_h += (sigmoid_val - 1) * outputMatrix[index]

        elif bit == '1':
            loss -= torch.log(1 - sigmoid_val)
            grad_out[index] = sigmoid_val * projection
            grad_h += sigmoid_val * outputMatrix[index]

    grad_in = torch.unsqueeze(grad_h, 0)

    return loss, grad_in, grad_out


def CBOW_NS(contextWords, inputMatrix, outputMatrix):
    K = outputMatrix.size(0)
    D = inputMatrix.size(1)

    projection = torch.zeros(D)
    for word in contextWords:
        projection += inputMatrix[word]

    scores = torch.squeeze(
        torch.mm(outputMatrix, torch.unsqueeze(projection, 1))
    )

    loss_scores = -scores
    loss_scores[0] *= -1
    log_sigmoids = torch.log(torch.sigmoid(loss_scores))

    loss = -torch.sum(log_sigmoids)

    grad_vh = torch.sigmoid(scores)
    grad_vh[0] -= 1

    grad_out = torch.mm(
        grad_vh.view(-1, 1),
        projection.view(1, -1)
    )

    grad_in = torch.mm(
        grad_vh.view(1, -1),
        outputMatrix
    )

    return loss, grad_in, grad_out


def choose_ns(num_ns, positive_index, numwords):
    count = 0
    ns = []
    while count < num_ns:
        ns_index = int(random() * numwords)
        if ns_index != positive_index:
            ns.append(ns_index)
            count += 1

    return ns

def choose_hs(code, tree):
    current_node = tree.root
    nodes_on_path = []

    for bit in code:
        nodes_on_path.append(current_node.index)
        if bit == '0':
            current_node = current_node.left
        elif bit == '1':
            current_node = current_node.right

    return nodes_on_path


def word2vec_trainer(input_seq, target_seq, numwords, codes, stats, mode="CBOW", NS=20, dimension=100, learning_rate=0.02, epoch=3, do_subsampling=False):
    W_in = torch.randn(numwords, dimension) / (dimension**0.5)

    if NS == 0:
        # Build Code Tree
        tree = BuildTree(codes.values())
        print(f"# of hierarchical softmax nodes: {tree.num_of_nodes}\n")
        W_out = torch.randn(tree.num_of_nodes, dimension) / (dimension**0.5)
    else:
        W_out = torch.randn(numwords, dimension) / (dimension**0.5)


    if mode == "SG":
        print(f"subsampling: {do_subsampling}")

    i=0
    losses=[]
    print("# of training samples")
    print(len(input_seq))
    print()
    stats = torch.LongTensor(stats)

    for _ in range(epoch):
        if do_subsampling and mode == "SG":
            subsample = subsampling(input_seq)

        for inputs, output in zip(input_seq, target_seq):
            i+=1
            if mode=="CBOW":
                if NS==0:
                    activated = choose_hs(codes[output], tree)
                    L, G_in, G_out = CBOW_HS(inputs, codes[output], W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in
                    W_out[activated] -= learning_rate*G_out

                else:
                    activated = [output] + choose_ns(NS, output, numwords)
                    L, G_in, G_out = CBOW_NS(inputs, W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in
                    W_out[activated] -= learning_rate*G_out

            elif mode=="SG":
                if do_subsampling and (inputs not in subsample):
                    continue

                if NS==0:
                    activated = choose_hs(codes[output], tree)
                    L, G_in, G_out = skipgram_HS(inputs, codes[output], W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in.squeeze()
                    W_out[activated] -= learning_rate*G_out
                else:
                    activated = [output] + choose_ns(NS, output, numwords)
                    L, G_in, G_out = skipgram_NS(inputs, W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in.squeeze()
                    W_out[activated] -= learning_rate*G_out

            else:
                print("Unknown mode : "+mode)
                exit()
            losses.append(L.item())
            if i%50000==0:
            	avg_loss=sum(losses)/len(losses)
            	print("Loss : %f" %(avg_loss,))
            	losses=[]

    return W_in, W_out


def main():
    parser = argparse.ArgumentParser(description='Word2vec')
    parser.add_argument('mode', metavar='mode', type=str,
                        help='"SG" for skipgram, "CBOW" for CBOW')
    parser.add_argument('ns', metavar='negative_samples', type=int,
                        help='0 for hierarchical softmax, the other numbers would be the number of negative samples')
    parser.add_argument('part', metavar='partition', type=str,
                        help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')
    args = parser.parse_args()
    mode = args.mode
    part = args.part
    ns = args.ns

    print("loading...")
    if part=="part":
        # 텍스트 수
        text = open('text8',mode='r').readlines()[0][:1000000]
    elif part=="full":
        text = open('text8',mode='r').readlines()[0]
    else:
        print("Unknown argument : " + part)
        exit()

    print("preprocessing...")
    corpus = text.split()
    stats = Counter(corpus)
    words = []

    for word in corpus:
        if stats[word]>4:
            words.append(word)
    vocab = set(words)

    w2i = {}
    w2i[" "]=0
    i = 1
    for word in vocab:
        w2i[word] = i
        i+=1
    i2w = {}
    for k,v in w2i.items():
        i2w[v]=k

    freqdict={}
    freqdict[0]=10
    for word in vocab:
        freqdict[w2i[word]]=stats[word]
    codedict = HuffmanCoding().build(freqdict)

    freqtable = [0,0,0]
    for k,v in stats.items():
        f = int(v**0.75)
        for _ in range(f):
            if k in w2i.keys():
                freqtable.append(w2i[k])

    print("build training set...")
    input_set = []
    target_set = []
    window_size = 5
    if mode=="CBOW":
        for j in range(len(words)):
            if j<window_size:
                input_set.append([0 for _ in range(window_size-j)] + [w2i[words[k]] for k in range(j)] + [w2i[words[j+k+1]] for k in range(window_size)])
                target_set.append(w2i[words[j]])
            elif j>=len(words)-window_size:
                input_set.append([w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[len(words)-k-1]] for k in range(len(words)-j-1)] + [0 for _ in range(j+window_size-len(words)+1)])
                target_set.append(w2i[words[j]])
            else:
                input_set.append([w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[j+k+1]] for k in range(window_size)])
                target_set.append(w2i[words[j]])
    if mode=="SG":
        for j in range(len(words)):
            if j<window_size:
                input_set += [w2i[words[j]] for _ in range(window_size*2)]
                target_set += [0 for _ in range(window_size-j)] + [w2i[words[k]] for k in range(j)] + [w2i[words[j+k+1]] for k in range(window_size)]
            elif j>=len(words)-window_size:
                input_set += [w2i[words[j]] for _ in range(window_size*2)]
                target_set += [w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[len(words)-k-1]] for k in range(len(words)-j-1)] + [0 for _ in range(j+window_size-len(words)+1)]
            else:
                input_set += [w2i[words[j]] for _ in range(window_size*2)]
                target_set += [w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[j+k+1]] for k in range(window_size)]

    print("Vocabulary size")
    print(len(w2i))
    print()

    numwords = len(w2i)
    use_subsample = True
    W_in ,_ = word2vec_trainer(input_set, target_set, numwords, codedict, freqtable, mode=mode, NS=ns, dimension=64, epoch=1, learning_rate=0.01, do_subsampling=use_subsample)

    emb = {}
    for index in range(numwords):
        emb[i2w[index]] = W_in[index]

    Analogical_Reasoning_Task(emb, output_name=f"{mode} {ns} {use_subsample} dim=64.txt")

main()
