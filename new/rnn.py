import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        num_layers=2,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(
            in_features=hidden_dim,
            out_features=output_dim,
        )

    def forward(self, x):

        # x shape:
        # (batch_size, sequence_length)

        embedded = self.embedding(x)

        # embedded shape:
        # (batch_size, sequence_length, embedding_dim)

        output, hidden = self.rnn(embedded)

        # hidden shape:
        # (num_layers, batch_size, hidden_dim)

        last_hidden = hidden[-1]

        # last_hidden shape:
        # (batch_size, hidden_dim)

        logits = self.fc(last_hidden)

        # logits shape:
        # (batch_size, output_dim)

        return logits