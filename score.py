###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch
from torch.autograd import Variable
import data
from utils import batchify
import torch.nn.functional as F
import os
import hashlib
import codecs
import csv

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='/proj/katinska/new_data',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN)')
parser.add_argument('--checkpoint', type=str, default='./RU12_best.pt',
                    help='model checkpoint to use')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model, criterion, optimizer = torch.load(f)


model.eval()
if args.model == 'QRNN':
    model.reset()

if args.cuda:
    model.cuda()
else:
    model.cpu()

fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)

ntokens = len(corpus.dictionary)
print('Dict length: ', ntokens)


def score(sentence):
    batch_size = 1
    tokens = sentence.split() + ['<eos>']
    idxs = [corpus.dictionary.get_index(x) for x in tokens]
    idxs = torch.LongTensor(idxs)
    # make it look as a batch of one element
    input = batchify(idxs, batch_size, args)
    # instantiate hidden states
    hidden = model.init_hidden(batch_size)
    output, hidden = model(input, hidden)
    logits = model.decoder(output)
    logProb = F.log_softmax(logits, dim=1)
    return [logProb[i][idxs[i+1]] for i in range(len((idxs))-1)]


if __name__ == '__main__':
    input_dir = './random_pos_sentences_02_2020'
    input_files = ['ma_clean_all.txt', 'annotated_er_all.txt', 'annotated_er_all_pos.txt']
    out_dir = './awd_results'
    for input_file in input_files:
        input = codecs.open(os.path.join(input_dir, input_file), 'r', encoding='utf-8')
        out_scores_file = input_file.replace('.txt', '_scores.csv')
        with codecs.open(out_scores_file, mode='w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['word', 'score'])
            for sent in input.readlines():
                print('Sentence length: ', len(sent))
                scores = score(sent)
                print('Score length: ', len(scores))
                tokens = sent.split()
                assert len(tokens) == len(scores)
                pairs = zip(tokens, scores)
                for pair in pairs:
                    writer.writerow([pair[0], pair[1].item()])