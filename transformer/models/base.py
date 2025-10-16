from typing import Dict

import torch
import torch.nn as nn


class Activation(nn.Module):
    def __init__(self, activation: str):
        """
        Define activation function through string input

        Parameters
        ----------
        activation: str
            Activation function to use in string
        """
        super(Activation, self).__init__()

        if activation.lower() == "relu":
            self.activation_func = nn.ReLU()
        elif activation.lower() == "leaky_relu":
            self.activation_func = nn.LeakyReLU()
        elif activation.lower() in {"gelu", "elu", "selu", "celu"}:
            self.activation_func = getattr(nn, activation.upper())()
        elif activation.lower() in {
            "tanh",
            "sigmoid",
            "hardshrink",
            "hardsigmoid",
            "tanhshrink",
            "softshrink",
            "softsign",
            "softplus",
        }:
            self.activation_func = getattr(nn, activation.capitalize())()
        else:
            raise ValueError(f"Invalid activation function: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation_func(x)


class ColumnEmbedding(nn.Module):
    def __init__(self, vocabulary: Dict[str, Dict[str, str]], embedding_dim: int):
        """
        Column embedding layer for categorical features

        Parameters
        ----------
        vocabulary: Dict[str, Dict[str, str]]
            Vocabulary of categorical features
            (e.g. {
                    'column_name_1':
                    {
                        'category_1_1': index_1,
                        'category_1_2': index_2,
                    },
                    'column_name_2':
                    {
                        'category_2_1': index_1,
                        'category_2_2': index_2,
                        'category_2_3': index_3,
                    }
                })
        embedding_dim: int
            Embedding dimension
        """
        super(ColumnEmbedding, self).__init__()
        self.embedding = nn.ModuleDict(
            {
                column: nn.Embedding(len(vocab), embedding_dim)
                for column, vocab in vocabulary.items()
            }
        )

    def forward(self, x: torch.Tensor, column: str) -> torch.Tensor:
        return self.embedding[column](x)
