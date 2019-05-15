from pythainlp import dict_word_tokenize
import marisa_trie
import os
import re

dir_path = os.getcwd()

stop_words = [line.strip() for line in open(dir_path + "/NLP_model/dict/dict_stopword.txt", 'r', encoding='utf-8')]
stop_trie = marisa_trie.Trie(stop_words)

dict_words = [line.strip() for line in open(dir_path + "/NLP_model/dict/cc_dict.txt", 'r', encoding='utf-8')]
dict_trie = marisa_trie.Trie(dict_words)

def token(sentence):
    sentence = sentence.replace('\n', '')
    sentence = sentence.replace('\t', '')
    sentence = re.sub(r'TOT|tot|ToT|WOW|wow|WoW|55+5|๕๕+๕', r'', sentence) # text emotion
    sentence = re.sub(r'[!]|[#]|[$]|[%]|[&]|[(]|[)]|[*]|[+]|[,]|[;]|[<]|[=]|[>]|[?][@]|[[]|[]]|[_]|[|]|[`]|[{]|[}]|[~]|["]', r' ', sentence)
    
    token = dict_word_tokenize(sentence, dict_trie, engine='newmm')
    point = 0
    while point < len(token):
        if token[point] in stop_trie:
            del token[point]
            continue
        else:
            point += 1
    return token