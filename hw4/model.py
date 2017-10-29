import torch
import torch.nn as nn
from torch.autograd import Variable

#Part 1 Uni-RNN
class RNNLM(nn.Module):
  def __init__(self, vocab_size):
    super(RNNLM, self).__init__()
    self.hidden_dim = 16 
    embedding_size = 32
    self.vocab_size = vocab_size
    #random word embedding
    self.embeddings = torch.rand(vocab_size, embedding_size)

  def forward(self, input_batch):
    sequence_length = input_batch.size()[0]
    batch_size = input_batch.size()[1]
    output_p = Variable(torch.zeros(sequence_length, batch_size, self.vocab_size), requires_grad=False)
    hidden_layer = Variable(torch.randn(batch_size, self.hidden_dim), requires_grad=True)


# TODO: Your implementation goes here
class BiRNNLM(nn.Module):
  def __init__(self, vocab_size):
    super(BiRNNLM, self).__init__()
    pass

  def forward(self, input_batch):
    pass
