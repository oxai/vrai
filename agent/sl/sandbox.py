import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import TransformerModel

args = {
"model":"Transformer",
"bptt": 5,
"lr":0.001,
"epochs":40,
"save":"test_run2",
"clip": 0.25,
"log_interval":10,
"dry_run": False,
"cuda":True
}

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
args = Struct(**args)

###############################################################################
# Load data
###############################################################################
#%%

# ideally would do something like use one of those discrete latent space autoencoders on the state space.

device = torch.device("cuda" if args.cuda else "cpu")
import numpy as np

obs = np.load("circling_box_obs.npy")
acts = np.load("circling_box_acts.npy")

n=10
discretized_obs = np.sum(np.floor(((obs+1)/2)*n)*np.array([1,n,n**2]),axis=1)
change_times = (np.where(np.diff(discretized_obs)>0)[0]) #max change time is 2000
states = discretized_obs[change_times]
change_times = n**3+change_times//100

state_keys = np.unique(states)
np.save("state_keys",state_keys)

thing = -np.ones(n**3)
thing[[int(x) for x in list(state_keys)]] = np.arange(state_keys.size)

processed_states = thing[[int(x) for x in list(states)]]

# len(states)
# combined = np.stack([change_times,states],axis=1)
# change_times[:10]
# states[:10]
# combined = combined.flatten()
# processed_states =

discretized_obs = np.expand_dims(processed_states,axis=1)
discretized_obs = torch.Tensor(discretized_obs).long()
discretized_obs.size(0)
# np.min(discretized_obs)
# np.max(discretized_obs)
discretized_obs_train = discretized_obs#[:4000]
discretized_obs_val = discretized_obs[2220:2222]
discretized_obs_test = discretized_obs[2222:]
len(discretized_obs)
# discretized_acts = np.sum(np.floor(acts*10)*np.array([1,10,100]),axis=1)
# discretized_acts += 1000
# np.unique(discretized_acts).size
# np.min(discretized_acts)
# np.max(discretized_acts)
# discretized_acts_train = discretized_acts[:1500]
# discretized_acts_test = discretized_acts[1500:]

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

batch_size=10
eval_batch_size = 10
train_data = batchify(discretized_obs_train, batch_size)
val_data = batchify(discretized_obs_val, eval_batch_size)
test_data = batchify(discretized_obs_test, eval_batch_size)

train_data.shape
# len(discretized_acts)
# len(np.where(discretized_acts != discretized_obs)[0])

#%%
###############################################################################
# Build the model
###############################################################################
# ntokens = n**3
# ntokens = n**3 + 20 #max change time is 2000
ntokens = state_keys.size
dropout = 0.0
nlayers=2
# nhid=16000
nhid=4000
# nhead=32
nhead=16
emsize=320
cuda=True


model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

criterion = nn.NLLLoss()


#%%
###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    if args.model != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        accuracy=0
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
            accuracy += ((torch.argmax(output,axis=1) == targets).float().sum())
    return total_loss / (len(data_source) - 1), accuracy/(len(data_source) - 1)

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
# count_parameters(model)

# torch.argmax(output,axis=1).shape
# targets.shape
# criterion(output, targets)
# data, targets = get_batch(train_data, 0)
# data.shape

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        # i=0
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        if args.model == 'Transformer':
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        # print(torch.all(model.decoder.weight.grad==0))
        # torch.all(model.encoder.weight.grad==0)

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # for p in model.parameters():
        #     p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            accuracy = ((torch.argmax(output,axis=1) == targets).float().sum())/len(output)
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | acc {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, accuracy))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break

    return cur_loss,accuracy

def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)

# Loop over epochs.
# lr = args.lr
best_val_loss = None
best_train_loss = None

import time
# At any point you can hit Ctrl + C to break out of training early.
import torch.optim as optim
# optimizer = optim.Adam(model.parameters(), lr=0.01)
args.epochs=900
args.log_interval=3
try:
    for epoch in range(1, args.epochs+1):
        optimizer = optim.SGD(model.parameters(), lr=lr)
        epoch_start_time = time.time()
        train_loss, train_acc = train()
        val_loss,acc = evaluate(val_data)
        # print('-' * 89)
        # print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
        #         'valid acc {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
        #                                    val_loss, acc))
        # print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_train_loss or train_loss < best_train_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_train_loss = train_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# output = model(train_data[:,:1])
# output.shape
# output = output.view(-1, ntokens)
# torch.argmax(output,axis=1)
#
# train_data[:,:1].t()

# np.add(*np.indices((m,m))//n)%2

#%%
len(discretized_obs)

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()

#%%
# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
#%%

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
