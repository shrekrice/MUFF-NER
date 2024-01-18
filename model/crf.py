import torch
import torch.nn as nn
import torch.autograd as autograd

S_TAG = -2
STOP = -1

def log_sum_exp(vec, dim):
    max_score, _ = torch.max(vec, dim)
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score.unsqueeze(dim)), dim))

class ConditionalRandomField(nn.Module):

    def __init__(self, num_tags, use_gpu):
        super(ConditionalRandomField, self).__init__()
        self.use_gpu = use_gpu
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.zeros(num_tags + 2, num_tags + 2))

        if self.use_gpu:
            self.transitions = self.transitions.cuda()

    def _calculate_partition(self, potentials, mask):
        batch_size, seq_len, num_tags = potentials.size()

        potentials = potentials.unsqueeze(2).expand(batch_size, seq_len, num_tags, num_tags)
        scores = potentials + self.transitions.view(1, 1, num_tags, num_tags).expand(batch_size, seq_len, num_tags, num_tags)

        partition = scores[0, 0, S_TAG, :].view(batch_size, num_tags, 1)

        for idx in range(1, seq_len):
            current_values = scores[:, idx, :, :] + partition.view(batch_size, num_tags, 1).expand(batch_size, num_tags, num_tags)
            current_partition = log_sum_exp(current_values, dim=1)
            mask_idx = mask[:, idx].view(batch_size, 1, 1)
            partition.masked_scatter_(mask_idx, current_partition.masked_select(mask_idx))

        final_partition = log_sum_exp(scores[:, -1, :, STOP], dim=1).sum()
        return final_partition, scores

    def _viterbi_decode(self, potentials, mask):
        batch_size, seq_len, num_tags = potentials.size()
        length_mask = mask.sum(dim=1).view(batch_size, 1).long()
        potentials = potentials.unsqueeze(2).expand(batch_size, seq_len, num_tags, num_tags)
        scores = potentials + self.transitions.view(1, 1, num_tags, num_tags).expand(batch_size, seq_len, num_tags, num_tags)

        back_pointers = []

        partition = scores[0, 0, S_TAG, :].view(batch_size, num_tags, 1)

        for idx in range(1, seq_len):
            current_values = scores[:, idx, :, :] + partition.view(batch_size, num_tags, 1).expand(batch_size, num_tags, num_tags)
            partition, current_back_pointer = torch.max(current_values, dim=1)
            partition_history.append(partition.unsqueeze(2))

            mask_idx = (1 - mask[:, idx]).view(batch_size, 1, 1)
            current_back_pointer.masked_fill_(mask_idx, 0)
            back_pointers.append(current_back_pointer)

        decode_indices = [autograd.Variable(torch.zeros(batch_size, dtype=torch.long))]
        pointer = back_pointers[-1][:, STOP]
        decode_indices.append(pointer)
        for idx in range(len(back_pointers) - 2, -1, -1):
            pointer = back_pointers[idx].gather(1, pointer.view(batch_size, 1))
            decode_indices.insert(1, pointer.squeeze(1))

        decode_indices = torch.stack(decode_indices, dim=1)
        return None, decode_indices

    def forward(self, potentials):
        path_score, best_path = self._viterbi_decode(potentials)
        return path_score, best_path

    def _score_sequence(self, scores, mask, tags):
        batch_size, seq_len, num_tags = scores.size()
        new_tags = autograd.Variable(torch.zeros(batch_size, seq_len, dtype=torch.long))

        if self.use_gpu:
            new_tags = new_tags.cuda()

        # Your code for scoring sequence goes here

        return new_tags
