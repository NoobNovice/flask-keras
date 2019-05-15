import pyuca
import sys
import re

data_in = [line.strip() for line in open(sys.argv[1], 'r', encoding='utf-8')]
data_sort = sorted(data_in, key = pyuca.Collator().sort_key)
text_out = open(sys.argv[1],"w", encoding='utf-8')
for data in data_sort:
    text_out.write(data+"\n")
text_out.close()