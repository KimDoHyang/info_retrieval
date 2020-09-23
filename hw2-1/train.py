import torch
from sampler import NegativeSampler

# Input:
# subwords: subword indices of the input center word
# input_matrix: weight matrix of input (type: torch.tensor(S,D)) (S: number of subwords)
# output_matrix: weight matrix of output (type: torch.tensor(K,D)) (K: number of activated outputs)
#
# Output:
# loss: loss value (type: torch.tensor(1))
# grad_in: gradient of input_matrix (type: torch.tensor(1,D))
# grad_out: gradient of output_matrix (type: torch.tensor(K,D))
def skipgram_ns(subwords, input_matrix, output_matrix):
    hidden = torch.sum(input_matrix[subwords], 0) # torch.tensor(D)

    scores = torch.squeeze(
        torch.mm(output_matrix, torch.unsqueeze(hidden, 1))
    ) # torch.tensor(K)

    loss_scores = -scores
    loss_scores[0] *= -1
    log_sigmoids = torch.log(torch.sigmoid(loss_scores))

    loss = -torch.sum(log_sigmoids)

    grad_vh = torch.sigmoid(scores) # torch.tensor(K)
    grad_vh[0] -= 1

    grad_out = torch.mm(
        grad_vh.view(-1, 1), # torch.tensor(K,1)
        hidden.view(1, -1) # torch.tensor(1,D)
    ) # torch.tensor(K,D)

    grad_in = torch.mm(
        grad_vh.view(1, -1), # torch.tensor(1,K)
        output_matrix #torch.tensor(K,D)
    ) # torch.tensor(1,D)

    return loss, grad_in, grad_out


def train_fasttext(corpus, ns_num=20, window_size=5, dimension=100, learning_rate=0.025, epoch=3, subsampling=True):
    # Xavier initialization of weight matrices
    W_in = torch.randn(corpus.number_of_subwords, dimension) / (dimension**0.5)
    W_out = torch.randn(corpus.number_of_words, dimension) / (dimension**0.5)

    if subsampling:
        print("subsampling: on")

    count=0
    losses=[]

    prob_weights = [freq ** 0.75 for freq in corpus.word_freqs]
    sampler = NegativeSampler(prob_weights)

    for epoch_index in range(epoch):
        print()
        print(f"Epoch #{epoch_index + 1}")
        input_seq, output_seq = corpus.generate_training_data(window_size, subsampling)
        print(f"# of training samples: {len(input_seq)}")
        print()

        print("Losses: ")
        for inputs, output in zip(input_seq, output_seq):
            count += 1

            activated = [output] + sampler.get_negative_samples(ns_num, output) # First element is the positive sample
            L, G_in, G_out = skipgram_ns(inputs, W_in, W_out[activated])
            W_in[inputs] -= learning_rate * G_in.squeeze()
            W_out[activated] -= learning_rate * G_out

            losses.append(L.item())

            if count%50000 == 0:
            	avg_loss = sum(losses) / len(losses)
            	print("%f" %(avg_loss), end=", ", flush=True)
            	losses=[]

    print("training finished!")
    return W_in, W_out
