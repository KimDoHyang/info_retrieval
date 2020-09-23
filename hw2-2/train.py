import torch
import hash

class TrainingData:
    def __init__(self, data):
        g2i, number_of_gram_indices = hash.generate_gram2index(data)

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
        print()

class ClassificationModel:
    def __init__(self, input_size, hidden_size, output_size):
        # Xavier initialization of weight matrices
        self.w_in = torch.randn(input_size, hidden_size) / (hidden_size**0.5)
        self.w_out = torch.randn(hidden_size, output_size) / (hidden_size**0.5)

    # train model with the given sample
    #
    # INPUT:
    # inputs: activated rows of input matrix
    # output: answer output index of the sample
    # learning_rate: learning rate to train model
    #
    # OUTPUT:
    # loss: loss value for the given sample
    def train(self, inputs, output, learning_rate=0.05):
        input_size = len(inputs)

        hidden = torch.sum(self.w_in[inputs], dim=0) / input_size  # (D)
        scores = torch.squeeze(
            torch.mm(hidden.view(1, -1), self.w_out)
        ) # (C)

        e_scores = torch.exp(scores)
        softmax = e_scores / torch.sum(e_scores)

        loss = -torch.log(softmax[output])

        softmax[output] -= 1

        grad_out = torch.mm(
            hidden.view(-1, 1), # (D,1)
            softmax.view(1, -1) # (1,C)
        )  # (D,C)

        grad_in = torch.mm(
            self.w_out, # (D,C)
            softmax.view(-1, 1) # (C,1)
        ).t() / input_size # (1,D)

        # Update weight matrices
        self.w_in[inputs] -= learning_rate * grad_in.squeeze()
        self.w_out -= learning_rate * grad_out

        return loss

    # Find output index and probability from the given inputs
    def classify(self, inputs):
        input_size = len(inputs)

        hidden = torch.sum(self.w_in[inputs], dim=0) / input_size  # (D)

        scores = torch.squeeze(
            torch.mm(hidden.view(1, -1), self.w_out)
        ) # (C)

        e_scores = torch.exp(scores)
        softmax = e_scores / torch.sum(e_scores)

        prob, index = softmax.max(dim=0)
        return index, prob

def train_classification_model(training_data, num_classes, dimension=64, learning_rate=0.05, epoch=3):

    classification_model = ClassificationModel(training_data.number_of_gram_indices, dimension, num_classes)

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

            if count % 5000 == 0:
                avg_loss = sum(losses) / len(losses)
                print(f'{avg_loss}\n')
                losses = []

    print()
    print("\nTraining finished!")
    return classification_model

