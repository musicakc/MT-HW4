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
    #random word embedding, this will be our input 
    self.embeddings = nn.Parameter(torch.randn(vocab_size, embedding_size), requires_grad=True)
    self.Wxh = nn.Parameter(torch.Tensor(32,16), requires_grad=True)
    self.Whh = nn.Parameter(torch.Tensor(16,16), requires_grad = True)
    self.Wyh = nn.Parameter(torch.Tensor(16, vocab_size), requires_grad =True)

  def softmax(self, x):
    calc = torch.exp(x)
    result = calc / torch.sum(calc)
    return result

  def forward(self, input_batch):
    sequence_length = input_batch.size()[0]
    batch_size = input_batch.size()[1]
    #check
    input_p = self.embeddings[input_batch.data, :].data
    output_p = Variable(torch.zeros(sequence_length, batch_size, self.vocab_size), requires_grad=False)
    hidden_layer = nn.Parameter(torch.randn(batch_size, self.hidden_dim), requires_grad=True)

    for i in range(sequence_length):
      input_temp = nn.Parameter(input_p[i, :, :])
      #calculate hidden state
      if (i == 0):
        hidden_c = torch.tanh(input_temp.mm(self.Wxh) + hidden_layer.mm(self.Whh))
      else:
        hidden_c = torch.tanh(input_temp.mm(self.Wxh) + last_hidden.mm(self.Whh))
      last_hidden = hidden_c
      op = hidden_c.mm(self.Wyh)
      #softmax function on op
      new_op = softmax(op)
      output_p.append(new_op)
    return output_p


# TODO: Your implementation goes here
class BiRNNLM(nn.Module):
  def __init__(self, vocab_size):
    super(BiRNNLM, self).__init__()
    pass

  def forward(self, input_batch):
    pass
