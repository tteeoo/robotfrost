import numpy as np
from random import randint
from torch import tensor
from torch.nn.functional import softmax

def predict(model, dataset, starter_text, next_words):
    """Generate output base on a starter_text using the forward pass."""

    # Set model to evaluation mode 
    model.eval()

    # Get starting text
    if starter_text == '':
        starter_text = dataset.index_to_word[randint(0, len(dataset.uniq_words))]

    # Initialize state
    words = starter_text.split(' ')
    state_h, state_c = model.init_state(len(words))

    # Iterate for amount of words to generate
    for i in range(0, next_words - len(words)):

        # Run forward pass
        x = tensor([[dataset.word_to_index[w] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        # Append the most likely word to the sentence 
        last_word_logits = y_pred[0][-1]
        p = softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])

    return words

def poem(model, dataset, starter_text, next_words):
    """Wrapper to predict, formats output."""

    # Call predict
    text = predict(model, dataset, starter_text, next_words)

    # Lower all-caps words
    for i in range(len(text)):
        if text[i].isupper():
            text[i] = text[i].lower()
            
    text = ' '.join(text)

    # Remove bad quotes and parens
    if text.count('(') != text.count(')'):
        text = text.replace('(', '')
        text = text.replace(')', '')
    if text.count('"') % 2 != 0:
        text = text.replace('"', '')

    # Separate sentences
    text = text.replace('.', '%%%%%%%%%%%').replace(';', '%%%%%%%%%%%').replace('_', '')
    output = [i for i in text.split('%%%%%%%%%%%') if i != '']
    output = [output[i][1:] for i in range(len(output)) if output[i][0] == ' ']
    output = [output[i][0].upper() + output[i][1:] for i in range(len(output))]

    for i in range(len(output)):
        for j in range(len(output[i])):
            if j == 0:
                continue
            if output[i][j].isupper() and (output[i][j-1] == ' '):
                a = list(output[i])
                a.insert(j, '\n')
                output[i] = ''.join(a)

    return '\n'.join(output)
