import torch
import torch.nn as nn


class BERTModel(nn.Module):
    """A lightweight transformer-based classifier for token ID sequences."""

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=2,
        num_layers=2,
        num_heads=4,
        max_length=200,
        dropout=0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=embedding_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.dropout = nn.Dropout(dropout)
        self.pooler = nn.Linear(embedding_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        """
        x shape: (batch_size, sequence_length)
        returns logits shape: (batch_size, output_dim)
        """
        batch_size, seq_len = x.shape

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        embedded = self.embedding(x) + self.position_embedding(positions)
        embedded = self.dropout(embedded)

        encoded = self.transformer(embedded)

        # Use the first token as the pooled representation for classification.
        pooled = encoded[:, 0, :]
        pooled = self.activation(self.pooler(pooled))
        logits = self.classifier(pooled)

        return logits
