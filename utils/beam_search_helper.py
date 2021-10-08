import torch
import torch.nn.functional as F
from model.TransModel import generate_square_subsequent_mask


class Beam:
    """
    The beam class for handling beam search.
    partly adapted from
    https://github.com/OpenNMT/OpenNMT-py/blob/195f5ae17f572c22ff5229e52c2dd2254ad4e3db/onmt/translate/beam.py
    There are some place which needs improvement:
    1. The prev_beam should be separated as prev_beam and beam_score.
    The prev_beam should be a tensor and beam_score should be a numpy array,
    such that the beam advance() method could speeds up.
    2. Do not support advance function like length penalty.
    3. If the beam is done searching, it could quit from further computation.
    In here, an eos is simply appended and still go through the model in next iteration.
    """

    def __init__(self, beam_size, device, start_symbol_ind, end_symbol_ind):
        self.device = device
        self.beam_size = beam_size
        self.prev_beam = [[torch.ones(1).fill_(start_symbol_ind).long().to(device), 0]]
        self.start_symbol_ind = start_symbol_ind
        self.end_symbol_ind = end_symbol_ind
        self.eos_top = False
        self.finished = []
        self.first_time = True

    def advance(self, word_probs, first_time):  # word_probs: (beam_size, ntoken) or (1, ntoken) for the first time.

        if self.done():
            # if current beam is done, just add eos to the beam.
            for b in self.prev_beam:
                b[0] = torch.cat([b[0], torch.tensor(self.end_symbol_ind).unsqueeze(0).to(self.device)])
            return

        # in first time, the beam need not to align with each index.
        if first_time:  # word_probs:(1, ntoken)
            score, index = word_probs.squeeze(0).topk(self.beam_size, 0, True, True)  # get the initial topk
            self.prev_beam = []
            for s, ind in zip(score, index):
                # initialize each beam
                self.prev_beam.append([torch.tensor([self.start_symbol_ind, ind]).long().to(self.device), s.item()])
                self.prev_beam = self.sort_beam(self.prev_beam)
        else:  # word_probs:(beam_size, ntoken)
            score, index = word_probs.topk(self.beam_size, 1, True, True)  # get topk
            current_beam = [[b[0].clone().detach(), b[1]] for b in self.prev_beam for i in range(self.beam_size)]
            # repeat each beam beam_size times for global score comparison, need to detach each tensor copied.
            i = 0
            for score_beam, index_beam in zip(score, index):  # get topk scores and corresponding index for each beam
                for s, ind in zip(score_beam, index_beam):
                    current_beam[i][0] = torch.cat([current_beam[i][0], ind.unsqueeze(0)])
                    # append current index to beam
                    current_beam[i][1] += s.item()  # add the score
                    i += 1

            current_beam = self.sort_beam(current_beam)  # sort current beam
            if current_beam[0][0][-1] == self.end_symbol_ind:  # check if the top beam ends with eos
                self.eos_top = True

            # check for eos node and added them to finished beam list.
            # In the end, delete those nodes and do not let them have child note.
            delete_beam_index = []
            for i in range(len(current_beam)):
                if current_beam[i][0][-1] == self.end_symbol_ind:
                    delete_beam_index.append(i)
            for i in sorted(delete_beam_index, reverse=True):
                self.finished.append(current_beam[i])
                del current_beam[i]

            self.prev_beam = current_beam[:self.beam_size]  # get top beam_size beam
            # print(self.prev_beam)

    def done(self):
        # check if current beam is done searching
        return self.eos_top and len(self.finished) >= 1

    def get_current_state(self):
        # get current beams
        # print(self.prev_beam)
        return torch.stack([b[0] for b in self.prev_beam])

    def get_output(self):
        if len(self.finished) > 0:
            # sort the finished beam and return the sentence with the highest score.
            self.finished = self.sort_beam(self.finished)
            return self.finished[0][0]
        else:
            self.prev_beam = self.sort_beam(self.prev_beam)
            return self.prev_beam[0][0]

    def sort_beam(self, beam):
        # sort the beam according to the score
        return sorted(beam, key=lambda x: x[1], reverse=True)


def beam_search(model, src, tokenizer, max_len, beam_size, start_token_id):
    start_symbol_ind = start_token_id
    end_symbol_ind = tokenizer.sep_token_id
    device = src.device  # src:(batch_size,T_in,feature_dim)
    batch_size = src.size()[0]
    memory = model.encode(src)  # memory:(T_mem,batch_size,nhid)
    # ys = torch.ones(batch_size, 1).fill_(start_symbol_ind).long().to(device)  # ys_0: (batch_size,T_pred=1)

    first_time = True

    beam = [Beam(beam_size, device, start_symbol_ind, end_symbol_ind) for _ in range(batch_size)]  # a batch of beams

    for i in range(max_len):
        # end if all beams are done, or exceeds max length
        if all((b.done() for b in beam)):
            break

        # get current input
        ys = torch.cat([b.get_current_state() for b in beam], dim=0).to(device).requires_grad_(False)

        # get input mask
        target_mask = generate_square_subsequent_mask(ys.size()[1]).to(device)
        out = model.decode(memory, ys, target_mask=target_mask)  # (T_out, batch_size, ntoken) for first time,
        # (T_out, batch_size*beam_size, ntoken) in other times
        # out = F.log_softmax(out[-1, :], dim=-1)  # (batch_size, ntoken) for first time,
        # (batch_size*beam_size, ntoken) in other times
        out = F.log_softmax(out.transpose(0,1)[-1, :], dim=-1)
        beam_batch = 1 if first_time else beam_size
        # in the first run, a slice of 1 should be taken for each beam,
        # later, a slice of [beam_size] need to be taken for each beam.
        for j, b in enumerate(beam):
            b.advance(out[j * beam_batch:(j + 1) * beam_batch, :], first_time)  # update each beam

        if first_time:
            first_time = False  # reset the flag
            # after the first run, the beam expands, so the memory needs to expands too.
            memory = memory.repeat_interleave(beam_size, dim=1)

    output = [b.get_output() for b in beam]
    return output
