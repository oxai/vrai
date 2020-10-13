###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch

# import data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
# parser.add_argument('--checkpoint', type=str, default='./model.pt',
parser.add_argument('--checkpoint', type=str, default='./test_run2',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
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

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)
model.eval()

# corpus = data.Corpus(args.data)
# ntokens = len(corpus.dictionary)
n=10
# ntokens=n**3

import numpy as np
state_keys = np.load("state_keys.npy")
ntokens = state_keys.size
#
# obs = np.load("circling_box_obs.npy")
# acts = np.load("circling_box_acts.npy")
#
# discretized_obs = np.sum(np.floor(((obs+1)/2)*n)*np.array([1,n,n**2]),axis=1)
# change_times = (np.where(np.diff(discretized_obs)>0)[0]) #max change time is 2000
# states = discretized_obs[change_times]

import numpy as np
is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
if not is_transformer_model:
    hidden = model.init_hidden(1)
input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
# input = torch.from_numpy(np.array([[651]])).long().to(device)

with open(args.outf, 'w') as outf:
    with torch.no_grad():  # no tracking history
        for i in range(args.words):
            if is_transformer_model:
                output = model(input, False)
                word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                input = torch.cat([input, word_tensor], 0)
                # if input.size(0)>39:
                input = input[1:]

            else:
                output, hidden = model(input, hidden)
                word_weights = output.squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                input.fill_(word_idx)

            # word = corpus.dictionary.idx2word[word_idx]
            word_idx = state_keys[word_idx]
            if word_idx>n**3:
                outf.write(str((word_idx.item()-n**3)*100) + "\n")
            else:
                word_idx = word_idx.item()
                xcoord = word_idx//n**2
                ycoord = (word_idx%n**2)//n
                zcoord = word_idx%n
                outf.write("["+str(2*xcoord/n-1)+";"+str(2*ycoord/n-1)+";"+str(2*zcoord/n-1)+"]" + "\n")



            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))
