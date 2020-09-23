import torch


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


def generate_gram2index(data):
    g2h = {}
    for title in data.titles:
        for gram in title:
            g2h[gram] = fnv32a(gram)

    for content in data.descriptions:
        for gram in content:
            g2h[gram] = fnv32a(gram)

    hashes = set([])
    for _, hash_value in g2h.items():
        hashes.add(hash_value)

    h2i = {}
    for index, hash_value in enumerate(hashes):
        h2i[hash_value] = index

    g2i = {}
    for gram, hash_value in g2h.items():
        g2i[gram] = h2i[hash_value]

    return g2i, len(hashes)


class TrainingData:
    def __init__(self, data):
        g2i, number_of_gram_indices = generate_gram2index(data)

        input_seq = []
        output_seq = []
        for index in range(data.length):
            input_seq.append(
                [g2i[gram] for gram in data.titles[index] + data.descriptions[index]]
            )

            output_seq.append(data.classes[index])

        self.g2i = g2i
        self.number_of_grams = len(g2i)
        self.number_of_gram_indices = number_of_gram_indices
        self.input_seq = input_seq
        self.output_seq = output_seq
        self.length = len(output_seq)
        print("Training data loaded!")
        print(f"# of n-grams: {self.number_of_grams}")
        print(f"# of hashed n-grams: {self.number_of_gram_indices}")
        print(f"# of training samples: {self.length}")


class Classification:
    def __init__(self, input_size, dimension, output_size):
        self.w_in = torch.randn(input_size, dimension) / (dimension**0.5)
        self.w_out = torch.randn(dimension, output_size) / (dimension**0.5)

    def train(self, inputs, output, learning_rate=0.02):
        input_size = len(inputs)

        hidden = torch.sum(self.w_in[inputs], dim=0) / input_size
        scores = torch.squeeze(
            torch.mm(hidden.view(1, -1), self.w_out)
        )

        e_scores = torch.exp(scores)
        softmax = e_scores / torch.sum(e_scores)
        loss = -torch.log(softmax[output])

        softmax[output] -= 1

        grad_out = torch.mm(
            hidden.view(-1, 1),
            softmax.view(1, -1)
        )

        grad_in = torch.mm(
            self.w_out,
            softmax.view(-1, 1)
        ).t() / input_size

        self.w_in[inputs] -= learning_rate * grad_in.squeeze()
        self.w_out -= learning_rate * grad_out

        return loss

    def classify(self, inputs):
        input_size = len(inputs)

        hidden_layer = torch.sum(self.w_in[inputs], dim=0) / input_size

        scores = torch.squeeze(
            torch.mm(hidden_layer.view(1, -1), self.w_out)
        )

        exp_scores = torch.exp(scores)
        softmax = exp_scores / torch.sum(exp_scores)

        prob, index = softmax.max(dim=0)
        return index, prob


def train_classification_model(training_data, num_classes, dimension=100, learning_rate=0.02, epoch=3):

    classification_model = Classification(training_data.number_of_gram_indices, dimension, num_classes)

    print("Train classification model...")

    count = 0
    losses = []
    for epoch_index in range(epoch):
        print()
        print(f"Epoch #{epoch_index + 1}")
        print("Losses: ")
        for inputs, output in zip(training_data.input_seq, training_data.output_seq):
            count += 1
            loss = classification_model.train(inputs, output, learning_rate)

            losses.append(loss.item())

            if count % 10000 == 0:
                avg_loss = sum(losses) / len(losses)
                print(f'{avg_loss}\n')
                losses = []

    print()
    print("\nTraining finished!")
    return classification_model

