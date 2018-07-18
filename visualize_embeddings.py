
import csv
import re
import numpy as np
import itertools
import pandas as pd
import gensim
import itertools
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def get_titles(book_titles):
    """ provides a list of all book titles """
    min_word_length = 3
    taboo_words = ["The", "What", "Other", "For", "You",
                   "That","And", "With"]

    alist = []
    for i, raw_title in enumerate(book_titles[1:]):
        tokens = raw_title[0].split(";")
        
        if len(tokens) < 2:
            print("line {}: {}".format(i, tokens))
        title = tokens[1].replace('"', '')
        words = [word.capitalize() for word in re.split(r'\W+', title)
                 if len(word) >= min_word_length]
        alist.append([word for word in words if word not in taboo_words])
    return alist

def titles2vec(w_model, titles):
    """ convert titles to n x m matrices where n is the size of the word2vec and
        m is the number of words in the title. """
    model = gensim.models.KeyedVectors.load_word2vec_format(w_model, binary=True)
    
    indices = []
    vecs = []
    title_indices = []
    for k,title in enumerate(titles):
        idx = [i for i,word in enumerate(title) if word in model.vocab]
        vec = [model[word][:,np.newaxis] for word in title if word in model.vocab]
        if len(vec) > 1:
            vecs.append(np.concatenate(vec,axis=1))
            indices.append(idx)
            title_indices.append(k)
    return vecs, indices, title_indices


with open('dataset/BX-Books.csv', 'rU') as fid:
     books_titles = list(csv.reader(fid))
     titles = get_titles(books_titles)
     titles = list(itertools.chain.from_iterable(titles))

vecs,indices,title_indices = titles2vec('GoogleNews-vectors-negative300.bin', titles)

obs = np.concatenate([title for title in vecs], axis=1).T

tsne = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
Y = tsne.fit_transform(obs[:300])

plt.scatter(Y[:, 0], Y[:, 1])
for label, x, y in zip(titles, Y[:, 0], Y[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
plt.show()
