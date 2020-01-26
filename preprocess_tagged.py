# -*- coding: utf-8 -*-

import os
import sys
import codecs

data_dir = '/Users/katinska/awd-lstm-lm/data'

if __name__ == '__main__':
    data_file = codecs.open(os.path.join(data_dir, 'tmp_tagged.txt'), 'r', encoding='utf-8')
    for line in data_file:
        print(line.strip())
        new_line = []
        line = line.split('\t')
        token, tag, lemma = line
        if tag == ',':
            tag = 'None'
        elif tag == '-':
            tag = 'None'
        if lemma == '<unknown>':
            lemma = '<unk>'


