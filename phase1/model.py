import torch
import torch.nn as nn
import torchvision

"""
This file is intended for the implementation of SuperGlue model, as explained in this paper:
https://arxiv.org/pdf/1911.11763.pdf
"""

ENCODER_DIM = 10


class keypoint_encoder(nn.Module):
    """
    encodes the input keypoint
    """
    def __init__(self, channels, position_length, descriptor_dim, encoder_dim):
        """

        :param channels: number of channels from the image
        :param position_length: length of vector representing position of a keypoint
        :param descriptor_dim: dimension of descriptor vector for a single key point
        """
        super(keypoint_encoder, self).__init__()
        self.linear_encoder = nn.Linear(channels + position_length + descriptor_dim,
                                        encoder_dim)

    def concatenate_input(self, channel, position, descriptor):
        """

        :param channel: has size (N, C)
        :param position: has size (N, position_length)
        :param descriptor: has size (N, descriptor_dim)
        :return:
        """
        # concatenate along dimension of one keypoint instance
        return torch.cat([channel, position, descriptor], dim=1)

    def forward(self, input):
        """

        :param input: a dictionary with "channels", "position", "descriptor", each respectively
            is a torch.Tensor with dimension (NC), (N, position_vector_dim), (N, descriptor_dim)
        :return:
        """
        # code for extracting channel values, position matrix and descriptor matrix remains unimplemented
        keypoint_channels = input["channels"]
        keypoint_position = input["position"]
        keypoint_descriptor = input["descriptor"]
        concatenated_input = self.concatenate_input(keypoint_channels,
                                                    keypoint_position, keypoint_descriptor)
        return self.linear_encoder(concatenated_input)

class multi_head_attention(nn.Module):
    """
    when predicting the "ith" output, given ith query and all keys for referencing,
    the weight/attention score/impact of jth key vector in making the prediction is calculated by using
    dot product attention.
    """
    def __init__(self, query_in, query_out, key_in, key_out):
        """

        :param query_in: encoded input vector dimension
        :param query_out:
        :param key_in: encoded referencing vector (can either be
        input vector for self-attention, or another different input vector for cross-attention)
        :param key_out:

        """
        super(multi_head_attention, self).__init__()
        self.Q = nn.Linear(query_in, query_out)
        self.K = nn.Linear(key_in, key_out)
        self.V = nn.Linear(key_in, key_out)

    def forward(self, query, key, value):
        """

        :param query: has size (N, encoder_size)
        :param key: has size (M, encoder_size)
        :param value: has size (M, encoder_size)
        :return:
        """
        Query = self.Q(query) # (N, encoder_size)
        Key = self.K(key) # (M, encoder_size)
        Value = self.V(value) # (M, encoder_size)
        dk = key.shape[1]
        # take softmax w.r.t dimension representing "KEY"'s weight for each query;
        weight = nn.Softmax(dim=1)(Query.matmul(Key.T) / torch.sqrt(dk)) # has dimension (N, M)
        # each (i, j) indicates for ith Query, the weight applied on jth Value is (i, j)'s value
        result = weight.matmul(Value) # has dimension (N, encoder_size)
        return result





