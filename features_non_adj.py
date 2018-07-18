""" Convert a list of book titles to a word2vec representation """
from __future__ import print_function, division

import csv
import pickle
import re
from sklearn.cluster import KMeans
import gensim
import numpy as np
import math
import itertools
from sklearn import mixture
from scipy.spatial.distance import cdist, pdist
import matplotlib.pyplot as plt


def get_titles(book_titles):
    """ provides a list of book titles with some pre-processing"""
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

def title_words(books):
    """provide original book titles"""
    alist = []
    for i, raw_title in enumerate(books[1:]):
        tokens = raw_title[0].split(";")
        alist.append(tokens[1].replace('"', ''))
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


def conversion_stats(titles, vecs):
    """ Keep track of empty and partial conversions. """
    empty = partial = full = 0
    for title, vec in zip(titles, vecs):
        if   len(vec) == 0:
            empty += 1
        elif vec.shape[1] != len(title):
            partial += 1
        else:
            full += 1

    print("Empty titles {}, partial {}, full {}".format(empty, partial, full))

def title_to_pair(titles):
    """get the ordered pair for each word in the book title"""
    pairs = []
    for title in titles:
        #title.append(None)
        pairs.append([(word,title[i+1]) for i,word in enumerate(title[:-1])])
    return pairs

def conditional_prob(pairs):
    """compute conditional probability between
        each two pairs of words base on frequency"""
    probs = {}
    pairs = list(itertools.chain.from_iterable(pairs))#convert everything to list
    for a,b in pairs:
        if a not in probs.keys():
            probs[a] = {}
        if b not in probs[a].keys():
            probs[a][b] = 0
        probs[a][b] += 1
    for a in probs.keys():
        a_prob = sum(freq for freq in probs[a].values())
        
    for b in probs.keys():
        b_prob = sum(freq for freq in probs[b].values())
        for b in probs[a].keys():
            probs[a][b] /= (a_prob*b_prob)
    return probs

def surprise(pairs,probs):
    """computer surprise base on lowest probability"""
    surprise = []
    for title in pairs:
        list = [math.log(probs[b][a],2) for (a,b) in title]
        index = np.argmin(list)
        surprise.append((index,list[index]))
    return surprise

def eblow(df, n):
    kMeansVar = [KMeans(n_clusters=k).fit(df) for k in range(1, n)]
    centroids = [X.cluster_centers_ for X in kMeansVar]
    k_euclid = [cdist(df, cent,'euclidean') for cent in centroids]
    dist = [np.min(ke, axis=1) for ke in k_euclid]
    wcss = [sum(d**2) for d in dist]
    tss = sum(pdist(df)**2)/df.shape[0]
    bss = tss - wcss
    
    plt.grid(True)
    plt.ylabel('percentage of variance explained')
    plt.xlabel('number of clusters')
    plt.title('Elbow for KMeans clustering')
    plt.plot(bss/tss*100,"b*-")
    plt.show()

def main():
    """ script entry point """
    with open('dataset/BX-Books.csv', 'rU') as fid:
        books_titles = list(csv.reader(fid))

    titles = get_titles(books_titles)

    # using the model trained on the Google News corpus, 300 dimensions.
    # Donwload from:
    # https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
    """vecs,indices,title_indices = titles2vec('GoogleNews-vectors-negative300.bin', titles)

    with open('converted_titles.pkl', 'wb') as fid:
        pickle.dump([vecs,indices,title_indices], fid, protocol=pickle.HIGHEST_PROTOCOL)"""

    with open('converted_titles.pkl', 'rb') as f:
         vecs,indices,title_indices = pickle.load(f)


    obs = np.concatenate([title for title in vecs], axis=1).T
    #obs = random.shuffle(obs)[:max_samples]
    #eblow(obs, 400)

    kmeans = KMeans(n_clusters=10, random_state=0).fit(obs)
    quantized_titles = [kmeans.predict(title.T) for title in vecs]
    pairs = title_to_pair(quantized_titles)
    probs = conditional_prob(pairs)
    book_titles = title_words(books_titles)
    surp = surprise(pairs,probs)
    for k, (i,prob) in enumerate(surp):
        title = book_titles[title_indices[k]]
        T = titles[title_indices[k]]
        word = T[indices[k][i]]
        next_word = T[indices[k][i+1]]
        print("{},({},{}): {}".format(title,word,next_word,prob))


if __name__ == '__main__':
    main()
