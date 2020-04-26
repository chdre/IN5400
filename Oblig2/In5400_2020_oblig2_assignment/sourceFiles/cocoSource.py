from torch import nn
import torch.nn.functional as F
import torch
import numpy as np


######################################################################################################################
class imageCaptionModel(nn.Module):
    def __init__(self, config):
        super(imageCaptionModel, self).__init__()
        """
        "imageCaptionModel" is the main module class for the image captioning network

        Args:
            config: Dictionary holding neural network configuration

        Returns (creates):
            self.Embedding  : An instance of nn.Embedding, shape[vocabulary_size, embedding_size]
            self.inputLayer : An instance of nn.Linear, shape[number_of_cnn_features, hidden_state_sizes]
            self.rnn        : An instance of RNN
            self.outputLayer: An instance of nn.Linear, shape[hidden_state_sizes, vocabulary_size]
        """
        self.config = config
        self.vocabulary_size        = config['vocabulary_size']
        self.embedding_size         = config['embedding_size']
        self.number_of_cnn_features = config['number_of_cnn_features']
        self.hidden_state_sizes     = config['hidden_state_sizes']
        self.num_rnn_layers         = config['num_rnn_layers']
        self.cell_type              = config['cellType']

        # ToDo
        self.Embedding = torch.nn.Embedding(self.vocabulary_size, self.embedding_size)

        self.inputLayer = torch.nn.Linear(self.number_of_cnn_features, self.hidden_state_sizes)

        self.rnn = RNN(self.embedding_size, self.hidden_state_sizes, self.num_rnn_layers, self.cell_type)

        self.outputLayer = torch.nn.Linear(self.hidden_state_sizes, self.vocabulary_size)
        return

    def forward(self, cnn_features, xTokens, is_train, current_hidden_state=None):
        """
        Args:
            cnn_features        : Features from the CNN network, shape[batch_size, number_of_cnn_features]
            xTokens             : Shape[batch_size, truncated_backprop_length]
            is_train            : "is_train" is a flag used to select whether or not to use estimated token as input
            current_hidden_state: If not None, "current_hidden_state" should be passed into the rnn module
                                  shape[num_rnn_layers, batch_size, hidden_state_sizes]

        Returns:
            logits              : Shape[batch_size, truncated_backprop_length, vocabulary_size]
            current_hidden_state: shape[num_rnn_layers, batch_size, hidden_state_sizes]
        """
        # ToDO
        # Get "initial_hidden_state" shape[num_rnn_layers, batch_size, hidden_state_sizes].
        # Remember that each rnn cell needs its own initial state.
        batch_size = xTokens.size()[0]
        if current_hidden_state == None:
            initial_hidden_state = torch.tanh(self.inputLayer(cnn_features))
            initial_hidden_states = torch.zeros((self.num_rnn_layers,
                                                batch_size,
                                                self.hidden_state_sizes))
            for i in range(self.num_rnn_layers):
                for j in range(batch_size):
                    initial_hidden_states[i, j, :] = initial_hidden_state[j, :]
        else:
            initial_hidden_states = current_hidden_state

        # use self.rnn to calculate "logits" and "current_hidden_state"'
        logits, current_state_out = self.rnn(xTokens,
                                            initial_hidden_states,
                                            self.outputLayer,
                                            self.Embedding,
                                            is_train=is_train)

        return logits, current_state_out

######################################################################################################################
class RNN(nn.Module):
    def __init__(self, input_size, hidden_state_size, num_rnn_layers, cell_type='RNN'):
        super(RNN, self).__init__()
        """
        Args:
            input_size (Int)        : embedding_size
            hidden_state_size (Int) : Number of features in the rnn cells (will be equal for all rnn layers)
            num_rnn_layers (Int)    : Number of stacked rnns
            cell_type               : Whether to use vanilla or GRU cells

        Returns:
            self.cells              : A nn.ModuleList with entities of "RNNCell" or "GRUCell"
        """
        self.input_size        = input_size # 32
        self.hidden_state_size = hidden_state_size # 64
        self.num_rnn_layers    = num_rnn_layers
        self.cell_type         = cell_type

        # ToDo
        # Your task is to create a list (self.cells) of type "nn.ModuleList" and
        # populated it with cells of type "self.cell_type".
        list = []
        if self.cell_type == 'RNN':
            list.append(RNNCell(self.hidden_state_size, self.input_size))
        else:
            list.append(GRUCell(self.hidden_state_size, self.input_size))

        for i in range(num_rnn_layers - 1):
            if self.cell_type == 'RNN':
                list.append(RNNCell(self.hidden_state_size, self.hidden_state_size))
            else:
                list.append(GRUCell(self.hidden_state_size, self.hidden_state_size))
        self.cells = nn.ModuleList(list)

        return


    def forward(self, xTokens, initial_hidden_state, outputLayer, Embedding, is_train=True):
        """
        Args:
            xTokens:        shape [batch_size, truncated_backprop_length]
            initial_hidden_state:  shape [num_rnn_layers, batch_size, hidden_state_size]
            outputLayer:    handle to the last fully connected layer (an instance of nn.Linear)
            Embedding:      An instance of nn.Embedding. This is the embedding matrix.
            is_train:       flag: whether or not to feed in the predicated token vector as input for next step

        Returns:
            logits        : The predicted logits. shape[batch_size, truncated_backprop_length, vocabulary_size]
            current_state : The hidden state from the last iteration (in time/words).
                            Shape[num_rnn_layers, batch_size, hidden_state_sizes]
        """
        if is_train==True:
            seqLen = xTokens.shape[1] #truncated_backprop_length
        else:
            seqLen = 40 #Max sequence length to be generated

        emb = Embedding(xTokens)
        logits = []

        if is_train:
            for i in range(seqLen): # Looping over all columns
                x = emb[:, i, :]
                for j in range(self.num_rnn_layers): # All layers inside each column
                    x = self.cells[j](x, initial_hidden_state[j, :, :])  # New state
                    initial_hidden_state[j, :, :] = x   # Updating state for next layer
                logits.append(outputLayer(x))
        else:
            x = emb[:, 0, :] # Network is trained
            for i in range(seqLen):
                for j in range(self.num_rnn_layers):
                    x = self.cells[j](x, initial_hidden_state[j, :, :])
                    initial_hidden_state[j, :, :] =  x
                output = outputLayer(x)
                logits.append(output)
                x = Embedding(torch.argmax(output, dim=1)) # Word with highest prob

        # Produce outputs
        logits        = torch.stack(logits, dim=1)
        current_state = initial_hidden_state

        return logits, current_state

########################################################################################################################
class GRUCell(nn.Module):
    def __init__(self, hidden_state_size, input_size):
        super(GRUCell, self).__init__()
        """
        Args:
            hidden_state_size: Integer defining the size of the hidden state of rnn cell
            inputSize: Integer defining the number of input features to the rnn

        Returns:
            self.weight_u: A nn.Parametere with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                           variance scaling with zero mean.

            self.weight_r: A nn.Parametere with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                           variance scaling with zero mean.

            self.weight: A nn.Parametere with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                         variance scaling with zero mean.

            self.bias_u: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero.

            self.bias_r: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero.

            self.bias: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero.

        Tips:
            Variance scaling:  Var[W] = 1/n
        """
        self.hidden_state_sizes = hidden_state_size

        # TODO:
        self.weight_u = nn.Parameter(
                        torch.from_numpy(
                            np.random.normal(
                                loc=0.0,
                                scale=1 / np.sqrt(hidden_state_size+input_size),
                                size=(hidden_state_size+input_size,
                                        hidden_state_size)
                    )
                ).float()
            )
        self.bias_u   = nn.Parameter(torch.zeros((1, hidden_state_size)))

        self.weight_r = nn.Parameter(
                        torch.from_numpy(
                            np.random.normal(
                                loc=0.0,
                                scale=1 / np.sqrt(hidden_state_size+input_size),
                                size=(hidden_state_size+input_size,
                                        hidden_state_size)
                    )
                ).float()
            )
        self.bias_r   = nn.Parameter(torch.zeros((1, hidden_state_size)))

        self.weight = nn.Parameter(
                        torch.from_numpy(
                            np.random.normal(
                                loc=0.0,
                                scale=1 / np.sqrt(hidden_state_size+input_size),
                                size=(hidden_state_size+input_size,
                                        hidden_state_size)
                    )
                ).float()
            )
        self.bias   = nn.Parameter(torch.zeros((1, hidden_state_size)))
        return

    def forward(self, x, state_old):
        """
        Args:
            x: tensor with shape [batch_size, inputSize]
            state_old: tensor with shape [batch_size, hidden_state_sizes]

        Returns:
            state_new: The updated hidden state of the recurrent cell. Shape [batch_size, hidden_state_sizes]

        """
        # TODO:
        concat = torch.cat((x, state_old), axis=1)

        u_gate  = torch.sigmoid(torch.mm(concat, self.weight_u) + self.bias_u)
        r_gate  = torch.sigmoid(torch.mm(concat, self.weight_r) + self.bias_r)

        concat2 = torch.cat((x, r_gate * state_old), axis=1)
        c_cell  = torch.tanh(torch.mm(concat2, self.weight) + self.bias)

        state_new = u_gate * state_old + (1 - u_gate) * c_cell
        return state_new

######################################################################################################################
class RNNCell(nn.Module):
    def __init__(self, hidden_state_size, input_size):
        super(RNNCell, self).__init__()
        """
        Args:
            hidden_state_size: Integer defining the size of the hidden state of rnn cell
            inputSize: Integer defining the number of input features to the rnn

        Returns:
            self.weight: A nn.Parameter with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                         variance scaling with zero mean.

            self.bias: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero.

        Tips:
            Variance scaling:  Var[W] = 1/n
        """
        self.hidden_state_size = hidden_state_size

        # TODO:
        self.weight = nn.Parameter(
                        torch.from_numpy(
                            np.random.normal(
                                loc=0.0,
                                scale=1 / np.sqrt(hidden_state_size+input_size),
                                size=(hidden_state_size+input_size,
                                        hidden_state_size)
                    )
                ).float()
            )
        self.bias   = nn.Parameter(torch.zeros((1, self.hidden_state_size)))
        return


    def forward(self, x, state_old):
        """
        Args:
            x: tensor with shape [batch_size, inputSize]
            state_old: tensor with shape [batch_size, hidden_state_sizes]

        Returns:
            state_new: The updated hidden state of the recurrent cell. Shape [batch_size, hidden_state_sizes]

        """
        # TODO:
        concat = torch.cat((x, state_old), axis=1)
        state_new = torch.tanh(torch.mm(concat, self.weight) + self.bias)
        return state_new

######################################################################################################################
def loss_fn(logits, yTokens, yWeights):
    """
    Weighted softmax cross entropy loss.

    Args:
        logits          : shape[batch_size, truncated_backprop_length, vocabulary_size]
        yTokens (labels): Shape[batch_size, truncated_backprop_length]
        yWeights        : Shape[batch_size, truncated_backprop_length]. Add contribution to the total loss only from words existing
                          (the sequence lengths may not add up to #*truncated_backprop_length)

    Returns:
        sumLoss: The total cross entropy loss for all words
        meanLoss: The averaged cross entropy loss for all words

    Tips:
        F.cross_entropy
    """
    eps = 0.0000000001 #used to not divide on zero
    # TODO:

    logits = logits.view(32*20, 100)
    yTokens = yTokens.view(32*20)
    yWeights = yWeights.view(32*20)

    loss = F.cross_entropy(input=logits, target=yTokens, reduction='none')
    arg = loss * yWeights

    sumLoss = torch.sum(arg)
    meanLoss = sumLoss / torch.sum(yWeights)

    return sumLoss, meanLoss
