"""
This file is intended for implementation of "superpoint" image keypoint detector,
following the idea from this article:
https://arxiv.org/pdf/1712.07629.pdf
"""

import torch
import torch.nn as nn
import torchvision

# below is an image for testing whether the model provides desired output, and for
# convenient bug detection.
test_image = torch.tensor(torch.arange(768 * 2).resize(2, 3, 16, 16), dtype=torch.float32) # batch, channel, H, W

KEYPOINT_THRESHOLD = 100
REMOVE_BORDER = 2
torch.manual_seed(1000)

class SuperPoint(nn.Module):
    """
    the input to this network is an image;
    the network process each image respectively first to extract possible keypoints from each image

    """
    def __init__(self, in_channel, descriptor_dim):
        super(SuperPoint, self).__init__()
        self.descriptor_dim = descriptor_dim
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activation = nn.ReLU()

        # encoder initialization
        self.encoder_channel = [in_channel * 2, in_channel * 4, in_channel * 8]
        self.conv1 = nn.Conv2d(in_channel, self.encoder_channel[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.encoder_channel[0], self.encoder_channel[1], kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv2d(self.encoder_channel[1], self.encoder_channel[2], kernel_size=3,
                               padding=1)
        self.encoder = nn.Sequential(self.conv1, self.maxpool, self.activation,
                                     self.conv2, self.maxpool, self.activation,
                                     self.conv3, self.maxpool, self.activation)

        # interest point decoder:
        self.conv_ipd = nn.Conv2d(self.encoder_channel[2], 65, kernel_size=3, padding=1) # 64 + 1, where "1" is "no keypoint"
        self.ipd_softmax = nn.Softmax(dim=1) # apply along "channel" dimension
        self.ipd = nn.Sequential(self.conv_ipd, self.ipd_softmax)

        # Descriptor decoder:
        self.conv_dd = nn.Conv2d(self.encoder_channel[2], descriptor_dim, kernel_size=3,
                                 padding=1)

    def bicubic_interpolation(self, input_image: torch.Tensor, output_size):
        """
        implements the mechanism for filling missing details when enlarging images
        to maintain clarity under high resolution;
        input image must have size (NCHW)
        """
        return nn.functional.interpolate(input_image, size=output_size, mode="bicubic")

    def extract_keypoint(self, score_matrix):
        """

        :param score_matrix: has size (N1HW), for evaluating each pixel's importance for one image
        :param input_image: has size NCHW
        :return: a masked image batch having size NCHW, where only keypoints are nonzero.
        """
        score_shape = score_matrix.shape
        # according to the paper, zero padding of scores for points on borders
        # can reduce noise. Can be achieved via tensor slicing and applying zero padding layer:
        zero_pad = nn.ZeroPad2d(REMOVE_BORDER)
        rb = REMOVE_BORDER # for notation convenience
        remove_border_score = zero_pad(score_matrix[:, :, rb:(score_shape[2] - rb),
                                                    rb:(score_shape[3] - rb)])

        # find "k" top key points
        flattened_score = remove_border_score.reshape(score_shape[0], score_shape[1],
                                               score_shape[2] * score_shape[3])
        top_k_keypoint = torch.topk(flattened_score, k=KEYPOINT_THRESHOLD, dim=-1)[1] # [1]: returns indices
        # create mask array
        mask = torch.zeros(flattened_score.shape)
        ones = torch.ones(flattened_score.shape)
        mask = mask.scatter(-1, top_k_keypoint, ones).reshape(score_shape) # mask has dimension N1HW
        return mask.nonzero(as_tuple=True) # returns coordinates instead of indices


    def forward(self, input_image: torch.Tensor):
        """
        :param input_image: has size (N, channel, H, W) or (channel, H, W)
        :return:
        """
        if len(input_image.shape) == 3: # assumed to be (CHW)
            input_image = input_image.unsqueeze(0) #add a dimension for batched input
        batch_size = input_image.shape[0]
        input_H = input_image.shape[2]
        input_W = input_image.shape[3]
        num_H_patches = input_H // 8
        num_W_patches = input_W // 8

        # encoded result
        encoded_result = self.encoder(input_image)

        # interest point decoder
        # provides score for each pixel, to determine whether it's a key point
        ipd_result = self.ipd(encoded_result).split(64, dim=1)[0] # dim=1: for removing the last "no keypoint" channel
        # according to the paper, each condensed (H, W)'s coordinate has 65 channels, representing
        # 8 x 8 grid plus one channel for "no keypoints". Thus can use "nn.Fold" method to
        # retain pixel-wise image score.
        ipd_result = ipd_result.reshape(batch_size, 64, num_H_patches * num_W_patches) # reshape result to feed as input of "Fold"
        fold = nn.Fold(output_size=(input_H, input_W), kernel_size=8, stride=8)
        ipd_score = fold(ipd_result)

        keypoint_mask = self.extract_keypoint(ipd_score) # indices, can be used for indexing
                                                        # and coordinate extraction
        coordinates = torch.cat([torch.unsqueeze(s, 0) for s in keypoint_mask],
                                dim=0).split(KEYPOINT_THRESHOLD, dim=1) # "cat" for forming coordinates,
                                        # split for each image's keypoints.
        key_point_score = ipd_score[keypoint_mask].split(KEYPOINT_THRESHOLD, dim=0)
        # descriptor decoder
        convolved_descriptor = self.conv_dd(encoded_result)
        output_size = (input_H, input_W)
        interpolated = self.bicubic_interpolation(convolved_descriptor, output_size)
        print("Interpolated size: " + str(interpolated.shape))
        normalized_descriptor = \
            nn.functional.normalize(interpolated)[keypoint_mask].split(KEYPOINT_THRESHOLD, dim=0)
        return coordinates, key_point_score, normalized_descriptor


model = SuperPoint(3, 10)
returned = model.forward(test_image)
a = 3