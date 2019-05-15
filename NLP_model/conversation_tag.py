import marisa_trie
import os
import re

dir_path = os.getcwd()

words = [line.strip() for line in open(dir_path + "/NLP_model/dict/dict_p1.txt", 'r', encoding='utf-8')]
p1_trie = marisa_trie.Trie(words)

words = [line.strip() for line in open(dir_path + "/NLP_model/dict/dict_p2.txt", 'r', encoding='utf-8')]
p2_trie = marisa_trie.Trie(words)

def tag(token):
    point = 0
    while point < len(token):
        if token[point] in p1_trie:
            token[point] = "p1"
        elif token[point] in p2_trie:
            token[point] = "p2"
        else:
            point += 1
    return token