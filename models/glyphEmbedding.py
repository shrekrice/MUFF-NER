from typing import List

import numpy as np
import torch
from torch import nn


class GlyphEmbedding(nn.Module):

    def __init__(self, font_npy_files: List[str]):
        super(GlyphEmbedding, self).__init__()

        font_arrays = [np.load(np_file).astype(np.float32) for np_file in font_npy_files]

        self.vocab_size, self.font_num, self.font_size = font_arrays[0].shape[0], len(font_arrays), \
                                                         font_arrays[0].shape[-1]

        font_array = np.stack(font_arrays, axis=1)
        flattened_size = self.font_size ** 2 * self.font_num
        self.embedding = nn.Embedding(self.vocab_size, flattened_size,
                                      _weight=torch.from_numpy(font_array.reshape([self.vocab_size, -1])))

    def forward(self, input_ids):
        return self.embedding(input_ids)
