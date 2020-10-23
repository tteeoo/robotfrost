import numpy as np
from random import randint
from torch import tensor
from torch.nn.functional import softmax

def predict(model, dataset, starter_text, next_words=100):
    """Function to generate output base on a starter_text using the forward pass."""

    # Get starting text
    if starter_text == '':
        starter_text = dataset.index_to_word[randint(0, len(dataset.uniq_words))]

    # Set model to evaluation mode 
    model.eval()

    # Initialize state
    words = starter_text.split(' ')
    state_h, state_c = model.init_state(len(words))

    # Iterate for amount of words to generate
    for i in range(0, next_words):

        # Run forward pass
        x = tensor([[dataset.word_to_index[w] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        # Append the most likely word to the sentence 
        last_word_logits = y_pred[0][-1]
        p = softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])

    return words
