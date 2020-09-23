from dataset import DataSet


def test_model(classes, g2i, classification_model, test_path, n_gram):
    test_data = DataSet(test_path, n_gram)

    output = open(f"test-result.txt", 'w')

    print("Testing start")
    correct_count = 0
    for index in range(test_data.length):
        answer = test_data.classes[index]
        inputs = []
        for gram in test_data.titles[index]:
            if gram in g2i:
                inputs.append(g2i[gram])

        for gram in test_data.descriptions[index]:
            if gram in g2i:
                inputs.append(g2i[gram])

        guess, prob = classification_model.classify(inputs)
        guess = guess.item()
        info = 'answer: {}, guess: {}, prob: {:.4} article: [{}]'.format(classes.class_name(answer), classes.class_name(guess), prob, test_data.titles_original[index])
        output.write(info + '\n')
        if answer == guess:
            correct_count += 1

    print("Test finished!")
    print(f"# of correct answer: {correct_count} / {test_data.length}")
    print()
    print("Accuracy = {:.4}".format(correct_count / test_data.length))
    output.write("Test finished!\n")
    output.write(f"# of correct answer: {correct_count} / {test_data.length}\n")
    output.write("\n")
    output.write("Accuracy = {:.4}\n".format(correct_count / test_data.length))
    output.close()
