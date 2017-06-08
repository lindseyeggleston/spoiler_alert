import numpy as np
import os

def file_to_text(filepath):
    if os.path.isfile(filepath):
        text = filepath.read().lower()
    elif os.path.isdir(filepath):
        with open(filepath + '/*.txt') as f:
            text = t.read().lower()
    return text

def text_to_vocab(text):
    pass

def text_to_sequence(text):
    pass

def create_minibatch(batchsize):
    pass
