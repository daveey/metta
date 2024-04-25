
import hashlib
from torch import nn
import numpy as np
import torch
import math

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Simple function to init layers
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

def make_nn_stack(
    input_size,
    output_size,
    hidden_sizes,
    nonlinearity,
    layer_norm=False,
    use_skip=False,
):
    """Create a stack of fully connected layers with nonlinearity"""
    sizes = [input_size] + hidden_sizes + [output_size]
    layers = []
    for i in range(1, len(sizes)):
        layers.append(nn.Linear(sizes[i - 1], sizes[i]))

        if i < len(sizes) - 1:
            layers.append(nonlinearity)

        if layer_norm and i < len(sizes) - 1:
            layers.append(nn.LayerNorm(sizes[i]))

    layers.append(nonlinearity)

    if use_skip:
        return SkipConnectionStack(layers)
    else:
        return nn.Sequential(*layers)

class SkipConnectionStack(nn.Module):
    def __init__(self, layers):
        super(SkipConnectionStack, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        skip_connections = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                if len(skip_connections) > 1:
                    x = x + skip_connections[-1]
                skip_connections.append(x)
            x = layer(x)
        return x


def stable_hash(s, mod=10000):
    """Generate a stable hash for a string."""
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod

def embed_strings(string_list, embedding_dim):
    return torch.tensor([
        embed_string(s, embedding_dim)
        for s in string_list
    ], dtype=torch.float32)

def embed_string(s, embedding_dim=128):
    # Hash the string using SHA-256, which produces a 32-byte hash
    hash_object = hashlib.sha256(s.encode())
    hash_digest = hash_object.digest()

    # Convert hash bytes to a numpy array of floats
    # This example simply takes the first 'embedding_dim' bytes and scales them
    byte_array = np.frombuffer(hash_digest[:embedding_dim], dtype=np.uint8)
    embedding = byte_array / 255.0  # Normalize to range [0, 1]

    # Ensure the embedding is the right size
    if len(embedding) < embedding_dim:
        # If the hash is smaller than the needed embedding size, pad with zeros
        embedding = np.pad(embedding, (0, embedding_dim - len(embedding)), 'constant')

    return embedding

def position_embeddings(width, height, embedding_dim=128):
    x = torch.linspace(-1, 1, width)
    y = torch.linspace(-1, 1, height)
    pos_x, pos_y = torch.meshgrid(x, y, indexing='xy')
    return torch.stack((pos_x, pos_y), dim=-1)

def sinusoidal_position_embeddings(width, height, embedding_dim=128):
    # Generate a grid of positions for x and y coordinates
    x = torch.linspace(-1, 1, width, dtype=torch.float32)
    y = torch.linspace(-1, 1, height, dtype=torch.float32)
    pos_x, pos_y = torch.meshgrid(x, y, indexing='xy')

    # Prepare to generate sinusoidal embeddings
    assert embedding_dim % 2 == 0, "Embedding dimension must be even."

    # Create a series of frequencies exponentially spaced apart
    freqs = torch.exp2(torch.linspace(0, math.log(embedding_dim // 2 - 1), embedding_dim // 2))

    # Apply sinusoidal functions to the positions
    embeddings_x = torch.cat([torch.sin(pos_x[..., None] * freqs), torch.cos(pos_x[..., None] * freqs)], dim=-1)
    embeddings_y = torch.cat([torch.sin(pos_y[..., None] * freqs), torch.cos(pos_y[..., None] * freqs)], dim=-1)

    # Combine x and y embeddings by summing, you could also concatenate or average
    embeddings = embeddings_x + embeddings_y

    # Add float embeddings
    if embedding_dim >= 2:
        embeddings[:,:,-2] = pos_x
        embeddings[:,:,-1] = pos_y

    return embeddings
