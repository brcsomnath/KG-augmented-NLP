import numpy as np
import re


_QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'
                       r'|^In article|^Quoted from|^\||^>)')


def strip_newsgroup_quoting(text):
    good_lines = [line for line in text.split('\n')
                  if not _QUOTE_RE.search(line)]
    return '\n'.join(good_lines)


def strip_newsgroup_header(text):
    _before, _blankline, after = text.partition('\n\n')
    return after


def strip_newsgroup_footer(text):

    lines = text.strip().split('\n')
    for line_num in range(len(lines) - 1, -1, -1):
        line = lines[line_num]
        if line.strip().strip('-') == '':
            break

    if line_num > 0:
        return '\n'.join(lines[:line_num])
    else:
        return text


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    #string = re.sub(r"[^A-Za-z0-9()@.,!?\'\`]", " ", string)
    string = re.sub(r"[^A-Za-z0-9()@.,!?\'\`]", " ", string)
    string = re.sub(r"\'s", " ", string)
    string = re.sub(r"\'m", " ", string)
    string = re.sub(r"\'ve", " ", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " ", string)
    string = re.sub(r"\'d", " ", string)
    string = re.sub(r"\'ll", " ", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r'\w*\d\w*', '', string)
    string = re.sub(r'\w*@\w*', '', string)
    string = re.sub(r"@", " ", string)
    string = re.sub(' +',' ',string)
    string = re.sub('\'','',string)
    string = re.sub('\`','',string)


    string = re.sub(r"\.", "\n", string)
    string = re.sub(r"\?", "\n", string)
    return string.strip().lower()


