import data

from utils import batchify, get_batch, repackage_hidden

data = '/data'

corpus = data.Corpus(data, preproc=True)

