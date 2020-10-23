from torch import nn
from torch import zeros

class RobotFrost(nn.Module):
    """Defines the architecture and functionality of the network."""

    def __init__(self, dataset):
        super(RobotFrost, self).__init__()
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 1

        # Embedding layer (translates words to a tensor)
        n_vocab = len(dataset.uniq_words)
        self.embedding = nn.Embedding(
            num_embeddings = n_vocab,
            embedding_dim = self.embedding_dim,
        )
        
        # LSTM layer (does the actual work)
        self.lstm = nn.LSTM(
            input_size = self.lstm_size,
            hidden_size = self.lstm_size,
            num_layers = self.num_layers,
            dropout = 0,
        )

        # Fully connected linear layer (gets out final output)
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state):
        """Defines our forward pass function."""

        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        """Returns the initial memory state of the LSTM."""

        return (zeros(self.num_layers, sequence_length, self.lstm_size),
            zeros(self.num_layers, sequence_length, self.lstm_size))
