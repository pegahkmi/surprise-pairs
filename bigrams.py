"""compute the surprising value for each book title base on lowest probability"""
from __future__ import print_function, division

import csv
import pickle
import re
import itertools
import gensim
import numpy as np


def get_titles(book_titles):
    """ provides a list of all book titles """
    min_word_length = 3
    taboo_words = ["The", "What", "Other", "for", "You",
                   "His","And", "Than", "Her", "Your"]
    
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

def title_to_pair(titles):
    """get the ordered pair for each word in the book title"""
    pairs = []
    for title in titles:
        #title.append(None)
        pairs.append([(word,title[i+1]) for i,word in enumerate(title[:-1])])
    return pairs

def conditional_prob_reverse(pairs):
    """compute conditional probability between 
    each two pairs of words base on frequency"""
    probs = {}
    pairs = list(itertools.chain.from_iterable(pairs))#convert everything to list
    for a,b in pairs:
        if b not in probs.keys():
            probs[b] = {}
        if a not in probs[b].keys():
           probs[b][a] = 0
        probs[b][a] += 1
    for b in probs.keys():
        b_prob = sum(freq for freq in probs[b].values())
        for a in probs[b].keys():
            probs[b][a] /= b_prob
    return probs

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
        
        for b in probs[a].keys():
            probs[a][b] /= a_prob
    return probs

def surprise(pairs,probs,probs_reverse):
    """computer surprise base on lowest probability"""
    surprise = []
    for title in pairs:
        list_reverse = [probs_reverse[b][a] for (a,b) in title]
        index_reverse = np.argmin(list_reverse)
        
        list = [probs[a][b] for (a,b) in title]
        index = np.argmin(list)
        
        if list[index] < list_reverse[index_reverse]:
           surprise.append((title[index][1],list[index]))
        else:
           surprise.append((title[index_reverse][0],list_reverse[index_reverse]))
    return surprise


def main():
    """ script entry point """
    with open('dataset/BX-Books.csv', 'rU') as fid:
        books_titles = list(csv.reader(fid))
    print((get_titles(books_titles[:])))

    titles = get_titles(books_titles[:])
    T = [x for x in titles if len(x)>1]
    pair = title_to_pair(T)
    probs_reverse = conditional_prob_reverse(pair)
    probs = conditional_prob(pair)
    surp = surprise(pair,probs,probs_reverse)

    print(surp[:100])


if __name__ == '__main__':
    main()
