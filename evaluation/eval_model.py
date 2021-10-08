import torch
from tqdm import tqdm
import torch.nn.functional as F
from evaluation.eval_metrics import evaluate_metrics
from utils.beam_search_helper import Beam, beam_search
import numpy as np


def eval_model(model, dataloader, tokenizer, config, device):
    model.eval()
    ground_truth = dataloader.dataset.filename_with_captions
    predicted = []
    beam_predicted = []
    for eval_data in tqdm(dataloader, desc='calculating metrics'):
        if config.w2v.enable or config.bert.auto_regressive or config.bert.input_id_all_not_empty:
            if not config.multisos.enable:
                y_hat_str = generate_text_auto_regressive(eval_data, tokenizer, config, model, device,
                                                          start_token_id=tokenizer.cls_token_id)
                y_hat_beam_str = generate_text_by_beam(eval_data, tokenizer, config, model,
                                                       start_token_id=tokenizer.cls_token_id,
                                                       device=device,
                                                       beam_size=config.decoder.beam_width)
            else:
                audio_embeddings = eval_data['audio_embedding']
                y_hat_ids = []
                y_hat_beam_ids = []
                for start_token_id in tokenizer.cls_token_ids:
                    y_hat_id = generate_text_auto_regressive(eval_data, tokenizer, config, model, device,
                                                             start_token_id=start_token_id,
                                                             return_token_id=True)
                    y_hat_beam_id = generate_text_by_beam(eval_data, tokenizer, config, model,
                                                          start_token_id=start_token_id,
                                                          device=device,
                                                          beam_size=config.decoder.beam_width,
                                                          return_token_id=True)
                    y_hat_ids.append(y_hat_id)
                    y_hat_beam_ids.append(y_hat_beam_id)
                y_hat_str = pick_result_base_on_captions(model, audio_embeddings, y_hat_ids, tokenizer,
                                                         max_length=config.decoder.max_length)
                y_hat_beam_str = pick_result_base_on_captions(model, audio_embeddings, y_hat_beam_ids, tokenizer,
                                                              max_length=config.decoder.max_length)

        else:
            raise ValueError("Temporarily only support auto regressive way!")
            # with torch.no_grad():
            #     y_hat = model(eval_data['audio_embedding'].to(device), eval_data['inputs'].to(device))
            # y_hat = torch.argmax(y_hat, dim=-1)
            # y_hat_str = tokenizer.batch_decode(y_hat.detach().cpu().numpy(), skip_special_tokens=True)
        filenames = eval_data['filename']
        for pred_caption, filename in zip(y_hat_str, filenames):
            predicted.append({
                'file_name': filename,
                'caption_predicted': pred_caption
            })
        for beam_pred_caption, filename in zip(y_hat_beam_str, filenames):
            beam_predicted.append({
                'file_name': filename,
                'caption_predicted': beam_pred_caption
            })
    metrics = evaluate_metrics(predicted, ground_truth)
    beam_metrics = evaluate_metrics(beam_predicted, ground_truth)
    return metrics, ground_truth, predicted, beam_metrics, beam_predicted


def generate_caption_file(filename, ground_truth, predicted):
    written_str = ""
    for gt, pred in zip(ground_truth, predicted):
        assert gt['file_name'] == pred['file_name'], "gt, pred not match!"
        written_str += "Captions for file {}\n".format(gt['file_name'])
        written_str += "	 Predicted caption: {}\n".format(pred['caption_predicted'])
        for caption_key in list(filter(lambda x: 'caption' in x, gt.keys())):
            written_str += "	 Original {}: {}\n".format(caption_key, gt[caption_key])
        written_str += "\n"
    with open(filename, 'w') as file:
        file.write(written_str)


# The eval_data must satisfy dataloader's format and in batch
def generate_text_auto_regressive(eval_data, tokenizer, config, model, device, start_token_id, return_token_id=False):
    model.eval()
    sample_num = len(eval_data['audio_embedding'])
    token_ids = [[start_token_id]] * sample_num
    for index in range(config.decoder.max_length):
        with torch.no_grad():
            inputs_ids = []
            attention_mask = []
            for iter_token_ids in token_ids:
                inputs_ids.append(iter_token_ids +
                                  [tokenizer.mask_token_id] * (config.decoder.max_length - len(iter_token_ids)))
                attention_mask.append([1] * len(iter_token_ids) +
                                      [0] * (config.decoder.max_length - len(iter_token_ids)))
            inputs_ids = torch.tensor(inputs_ids)
            attention_mask = torch.tensor(attention_mask)
            y_hat = model(eval_data['audio_embedding'].to(device), inputs_ids.to(device),
                          attention_mask=attention_mask.to(device))
            y_hat_ids = torch.argmax(y_hat, dim=-1)[:, index].reshape(sample_num, -1)
            y_hat_ids = y_hat_ids.detach().cpu().numpy().tolist()
            token_ids = [x + y for x, y in zip(token_ids, y_hat_ids)]
    for index, iter_token_ids in enumerate(token_ids):
        if tokenizer.sep_token_id in iter_token_ids:
            end_index = iter_token_ids.index(tokenizer.sep_token_id)
            iter_token_ids = iter_token_ids[:end_index]
            token_ids[index] = iter_token_ids
    if return_token_id:
        return token_ids
    y_hat_str = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    return y_hat_str


def generate_text_by_beam(eval_data, tokenizer, config, model, start_token_id, device='cuda',
                          beam_size=2, skip_special_tokens=True, return_token_id=False):
    model.eval()
    with torch.no_grad():
        src = eval_data['audio_embedding']
        y_hat = beam_search(model, src.to(device), tokenizer, config.decoder.max_length,
                            beam_size, start_token_id=start_token_id)
    y_hat_cpu = [y.detach().cpu().numpy().tolist() for y in y_hat]
    if return_token_id:
        return y_hat_cpu
    y_hat_str = tokenizer.batch_decode(y_hat_cpu, skip_special_tokens=skip_special_tokens)
    return y_hat_str


def padding_candidate(candidate, tokenizer, max_length=35, truncate=True):
    list_candidate = candidate.tolist()
    padded_candidate = []
    for sentence in list_candidate:
        if len(sentence) < max_length:
            sentence += [tokenizer.sep_token_id] * (max_length - len(sentence))
        elif truncate:
            sentence = sentence[:max_length]
        padded_candidate.append(sentence)
    return padded_candidate


def pick_result_base_on_captions(model, audio_embeddings, y_hat_ids, tokenizer, max_length=35):
    targets = []
    scores = []
    candidates = np.asarray(y_hat_ids)
    padded_candidates = []
    for candidate in candidates:
        padded_candidate = padding_candidate(candidate, tokenizer, max_length=max_length)
        with torch.no_grad():
            model.eval()
            _, score = model(audio_embeddings.to('cuda'), torch.tensor(padded_candidate).to('cuda'),
                             selection_result=True)
            score = torch.nn.Softmax(dim=-1)(score)
        scores.append(score.detach().cpu().numpy().tolist())
        padded_candidates.append(padded_candidate)
    scores = np.asarray(scores)
    prob1_scores = scores[:, :, 1]
    for batch_index in range(prob1_scores.shape[1]):
        best_index = np.argmax(prob1_scores[:, batch_index])
        targets.append(padded_candidates[best_index][batch_index])
        # targets.append(candidates[best_index, batch_index, :])
    target_strs = tokenizer.batch_decode(targets, skip_special_tokens=True)
    return target_strs
