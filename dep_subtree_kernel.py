#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: raghebal-ghezi
updated: 24.4.2024
"""
import nltk
from nltk.parse import DependencyGraph
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample sentences in dependency parse tree format
dep_parse_tree_str1 = """
1\tThe\t_\tDET\tDT\t_\t3\tdet\t_\t_
2\tbad\t_\tADJ\tJJ\t_\t3\tamod\t_\t_
3\tresearcher\t_\tNOUN\tNN\t_\t4\tnsubj\t_\t_
4\tshreds\t_\tVERB\tVBZ\t_\t0\troot\t_\t_
5\tthe\t_\tDET\tDT\t_\t7\tdet\t_\t_
6\timportant\t_\tADJ\tJJ\t_\t7\tamod\t_\t_
7\tpaper\t_\tNOUN\tNN\t_\t4\tdobj\t_\t_
8\t.\t_\tPUNCT\t.\t_\t4\tpunct\t_\t_
"""

dep_parse_tree_str2 = """
1\tThe\t_\tDET\tDT\t_\t3\tdet\t_\t_
2\tangry\t_\tADJ\tJJ\t_\t3\tamod\t_\t_
3\tdog\t_\tNOUN\tNN\t_\t4\tnsubj\t_\t_
4\tate\t_\tVERB\tVBD\t_\t0\troot\t_\t_
5\tthe\t_\tDET\tDT\t_\t7\tdet\t_\t_
6\tjuicy\t_\tADJ\tJJ\t_\t7\tamod\t_\t_
7\tsteak\t_\tNOUN\tNN\t_\t4\tdobj\t_\t_
8\t.\t_\tPUNCT\t.\t_\t4\tpunct\t_\t_
"""

dep_parse_tree_str3 = """
1\tThis\t_\tDET\tDT\t_\t2\tdet\t_\t_
2\tis\t_\tVERB\tVBZ\t_\t0\troot\t_\t_
3\tyour\t_\tPRP$\tPRP$\t_\t4\tposs\t_\t_
4\tmission\t_\tNOUN\tNN\t_\t2\tattr\t_\t_
5\tshould\t_\tAUX\tMD\t_\t2\taux\t_\t_
6\tyou\t_\tPRP\tPRP\t_\t7\tnsubj\t_\t_
7\tchoose\t_\tVERB\tVB\t_\t5\tccomp\t_\t_
8\tto\t_\tPART\tTO\t_\t9\tmark\t_\t_
9\taccept\t_\tVERB\tVB\t_\t7\txcomp\t_\t_
10\t.\t_\tPUNCT\t.\t_\t2\tpunct\t_\t_
"""

def extract_dependency_features(dep_parse_tree_str):
    dep_graph = DependencyGraph(dep_parse_tree_str, top_relation_label='root')
    features = {}

    for token_id, token_info in dep_graph.nodes.items():
        if token_id == 0 or token_info['word'] is None:
            continue
        head_word = token_info['word']
        dependent_words = [dep_graph.nodes[dep]['word'] for dep, dep_info in dep_graph.nodes.items() if dep_info['head'] == int(token_id)]
        features[head_word] = dependent_words

    return features

def binary_tree_kernel(features1, features2):
    vectorizer = TfidfVectorizer()
    corpus = [' '.join([' '.join([head] + dependents) for head, dependents in features1.items()]),
              ' '.join([' '.join([head] + dependents) for head, dependents in features2.items()])]
    
    vectors = vectorizer.fit_transform(corpus)
    similarity_matrix = cosine_similarity(vectors)
    
    return similarity_matrix[0, 1]

# Extract dependency features from sentences
features1 = extract_dependency_features(dep_parse_tree_str1)
features2 = extract_dependency_features(dep_parse_tree_str2)
features3 = extract_dependency_features(dep_parse_tree_str3)

# Compute similarity
similarity12 = binary_tree_kernel(features1, features2)
similarity13 = binary_tree_kernel(features1, features3)

# Print results
print("Features Sentence 1:", features1)
print("Features Sentence 2:", features2)
print("Features Sentence 3:", features3)

print("Binary tree kernel similarity between Sentence 1 and Sentence 2:", similarity12)
print("Binary tree kernel similarity between Sentence 1 and Sentence 3:", similarity13)
