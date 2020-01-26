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

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='/proj/katinska/new_data',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN)')
parser.add_argument('--checkpoint', type=str, default='./RU10.pt',
                    help='model checkpoint to use')
# parser.add_argument('--outf', type=str, default='generated.txt',
#                     help='output file for generated text')
# parser.add_argument('--words', type=int, default='1000',
#                     help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
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

# corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
# hidden = model.init_hidden(1)
# input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
# if args.cuda:
#     input.data = input.data.cuda()


def score(self, sentence):
    batch_size = 1
    tokens = sentence.split() + ['<eos>']
    idxs = [self.dictionary.get_index(x) for x in tokens]
    idxs = torch.LongTensor(idxs)
    # make it look as a batch of one element
    input = batchify(idxs, batch_size, args)
    # instantiate hidden states
    hidden = self.model.initHidden(batchSize=1)
    output, hidden = self.model(input, hidden)
    logits = self.model.decoder(output)
    logProba = F.log_softmax(logits, dim=1)
    return sum([logProba[i][idxs[i+1]] for i in range(len((idxs))-1)])


sentence = 'Мама мыла раму'
print('Score: ', score(sentence))



# with open(args.outf, 'w') as outf:
#     for i in range(args.words):
#         output, hidden = model(input, hidden)
#         word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
#         word_idx = torch.multinomial(word_weights, 1)[0]
#         input.data.fill_(word_idx)
#         word = corpus.dictionary.idx2word[word_idx]
#
#         outf.write(word + ('\n' if i % 20 == 19 else ' '))
#
#         if i % args.log_interval == 0:
#             print('| Generated {}/{} words'.format(i, args.words))

