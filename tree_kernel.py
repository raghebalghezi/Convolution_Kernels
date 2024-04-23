#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: raghebal-ghezi
updated: 24.4.2024
"""
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

s1 = nltk.Tree.fromstring("(S (NP (DT The) (JJ bad) (NN researcher)) (VP (VBZ shreds) (NP (DT the) (JJ important) (NN paper))) (.) )")
s2 = nltk.Tree.fromstring("(S (NP (DT The) (JJ angry) (NN dog)) (VP (VBD ate) (NP (DT the) (JJ juicy) (NN steak))) (.) )")
s3 = nltk.Tree.fromstring("(S (NP (DT This)) (VP (VBZ is) (NP (PRP$ your) (NN mission)) (SBAR (IN should) (S (NP (PRP you)) (VP (VB choose) (S (VP (TO to) (VP (VB accept))))))) (.)))")


def extract_subtrees(tree):
    subtrees = [str(prod) for prod in tree.productions()]
    return subtrees


def binary_tree_kernel(tree1, tree2):
    vectorizer = TfidfVectorizer().fit([tree1, tree2])
    features1 = vectorizer.transform([tree1]).toarray()
    features2 = vectorizer.transform([tree2]).toarray()
    return cosine_similarity(features1, features2)

s1 = ' '.join(extract_subtrees(s1))
s2 = ' '.join(extract_subtrees(s2))

print("Sentence 1:", s1)
print("Sentence 2:", s2)

print("Binary tree kernel similarity:", binary_tree_kernel(s1, s2)[0][0])
