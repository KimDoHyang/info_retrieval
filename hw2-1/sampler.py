import random


# Randomly choose negative samples according to given prob weights
class NegativeSampler:
    def __init__(self, prob_weights):
        total_weight = sum(prob_weights)
        bounds = []
        acc = 0
        for (index, weight) in enumerate(prob_weights):
            bounds.append(acc)
            acc += weight / total_weight
        bounds.append(1.0)

        self.number_of_candidates = len(prob_weights)
        self.bounds = bounds


    def get_negative_samples(self, number_of_ns, positive_index):
        count = 0
        negative_samples = []
        while count < number_of_ns:
            new_sample = self.choose_one()
            if new_sample == positive_index:
                continue
            negative_samples.append(new_sample)
            count += 1

        return negative_samples


    # Randomly choose one sample according to prob weights
    def choose_one(self):
        value = random.random()
        leftmost = 0
        rightmost = self.number_of_candidates - 1

        # Binary search to find upper bound about the random value
        while leftmost <= rightmost:
            mid = (leftmost + rightmost) // 2
            if self.bounds[mid] > value:
                rightmost = mid - 1
            elif self.bounds[mid] <= value:
                upper_bound = mid
                leftmost = mid + 1

        return upper_bound
