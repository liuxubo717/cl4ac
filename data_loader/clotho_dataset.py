import csv
from functools import reduce
import numpy as np
import torch
from cachetools import LRUCache, cached
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
from config_loader.config import get_config
from data_loader.logmel_loader import LogmelLoader
from data_loader.sample import Sample
import random


def read_cloth_file(filename, audio_loader=None):
    all_captions = []
    filename_with_captions = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            name_with_captions = {'file_name': row[0]}
            for index, caption in enumerate(row[1:]):
                audio_caption = {'audio': row[0], 'caption': caption, 'caption_index': index}
                if audio_loader is not None:
                    audio_caption['audio_embedding'] = audio_loader.get_embedding(row[0])
                all_captions.append(audio_caption)
                name_with_captions['caption_{}'.format(index + 1)] = caption
                name_with_captions['caption_index'] = index
            filename_with_captions.append(name_with_captions)
    return all_captions, filename_with_captions


class ClothoDataset(Dataset):
    def __init__(self, caption_path, config, tokenizer=None, is_train=True):
        self.config = config
        self.caption_path = caption_path
        self.is_train = is_train
        self.audio_loader = LogmelLoader(h5_path=config.dataset.audio_h5_path,
                                         wav_name_path=config.dataset.wave_name_path)
        if is_train:
            self.audio_captions, self.filename_with_captions = read_cloth_file(caption_path, self.audio_loader)
        else:
            self.audio_captions, self.filename_with_captions = read_cloth_file(caption_path)
        self.auto_regressive = config.bert.auto_regressive
        if config.bert.auto_regressive and is_train:
            print("Generating auto regressive training dataset")
            self.audio_captions_auto_regressive = []
            if tokenizer is None:
                raise ValueError("Need to have a tokenizer for pre-processing")
            for audio_caption in tqdm(self.audio_captions):
                audio = audio_caption['audio']
                caption = audio_caption['caption']
                caption_index = None
                if config.multisos.enable:
                    caption_index = audio_caption['caption_index']
                tokenized = tokenizer(caption, padding='max_length', max_length=config.decoder.max_length,
                                      caption_index=caption_index)
                tokenized_no_padding = tokenizer(caption, padding=False, caption_index=caption_index)
                for index in range(len(tokenized_no_padding['input_ids']) - 1):
                    inputs = tokenized['input_ids'][:index + 1]
                    targets = tokenized['input_ids'][1:index + 2]
                    attention_mask = [1] * len(inputs) + [0] * (config.decoder.max_length - len(inputs))
                    inputs = inputs + [tokenizer.mask_token_id] * (config.decoder.max_length - len(inputs))
                    targets = targets + [tokenizer.mask_token_id] * (config.decoder.max_length - len(targets))
                    self.audio_captions_auto_regressive.append({'audio': audio, 'caption': caption,
                                                                'inputs': inputs, 'targets': targets,
                                                                'attention_mask': attention_mask,
                                                                'caption_index': caption_index})
        self.random_sample_list = self.audio_captions.copy()

    def get_word_frequency(self, tokenizer):
        all_captions = [audio_caption['caption'] for audio_caption in self.audio_captions]
        tokenized = tokenizer(all_captions, add_special_tokens=False)
        tokenized = tokenized['input_ids']
        tokenized = reduce(lambda x, y: x + y, tokenized)
        uniq_tokens = list(set(tokenized))
        ratio = {}
        total = len(tokenized)
        tokenized = np.asarray(tokenized)
        for token in uniq_tokens:
            number = np.count_nonzero(tokenized == token)
            # make weight softer
            ratio[token] = math.log(1 / (number / (total + 1)))
        return ratio

    def __len__(self):
        if self.random_sample_list is not None:
            random.shuffle(self.random_sample_list)
        if self.is_train and self.auto_regressive:
            return len(self.audio_captions_auto_regressive)
        if self.is_train:
            return len(self.audio_captions)
        else:
            return len(self.filename_with_captions)

    def choose_negative_by_audio_name(self, audio_name, k=1):
        filter_iter = filter(lambda x: x['audio'] != audio_name, self.random_sample_list)
        results = []
        for result in filter_iter:
            results.append(result)
            if len(results) > k:
                return results
        return results

    def __getitem__(self, idx):
        sample = Sample()
        if self.is_train and self.auto_regressive:
            audio_caption = self.audio_captions_auto_regressive[idx]
            caption_index = audio_caption['caption_index']
            audio = audio_caption['audio']
            caption = audio_caption['caption']
            sample.inputs = audio_caption['inputs']
            sample.targets = audio_caption['targets']
            sample.attention_mask = audio_caption['attention_mask']
        elif self.is_train:
            audio_caption = self.audio_captions[idx]
            caption_index = audio_caption['caption_index']
            audio = audio_caption['audio']
            caption = audio_caption['caption']
            sample.attention_mask = None
        else:
            audio_caption = self.filename_with_captions[idx]
            caption_index = audio_caption['caption_index']
            audio = audio_caption['file_name']
            caption = ""
            sample.attention_mask = None
        sample.filename = audio
        audio_embedding = self.audio_loader.get_embedding(audio) if 'audio_embedding' not in audio_caption.keys() \
            else audio_caption['audio_embedding']
        sample.audio_embedding = audio_embedding
        sample.caption_text = caption
        sample.caption_index = caption_index
        if self.config.auxiliary_task.selection_loss:
            sample.negative_caption_text = self.choose_negative_by_audio_name(sample.filename)[0]['caption']
        return sample


def collate_fn(sample_list):
    to_be_flattened = ['audio_embedding', 'caption_text', 'filename', 'caption_index',
                       'inputs', 'targets', 'attention_mask', 'negative_caption_text']
    data = {}
    for key in to_be_flattened:
        if key not in sample_list[0].keys():
            continue
        if sample_list[0][key] is None:
            continue
        flatten_samples = [sample[key] for sample in sample_list]
        if flatten_samples[-1].__class__ == str:
            data[key] = flatten_samples
        else:
            data[key] = torch.tensor(flatten_samples)
    return data


def collate_fn_with_tokenizer(tokenizer, max_length, input_id_as_empty=False, input_id_all_not_empty=False,
                              multisos=False):
    def build_collate_fn(sample_list):
        data = collate_fn(sample_list)
        caption_index = None
        if multisos:
            caption_index = data['caption_index']
        if 'inputs' in data.keys() and 'targets' in data.keys():
            return data
        # Because we will discard [CLS] in targets, so we add max_length by 1
        add_special_tokens = True
        if input_id_as_empty:
            add_special_tokens = True
        tokenized = tokenizer(data['caption_text'], return_tensors='pt', add_special_tokens=add_special_tokens,
                              padding='max_length', max_length=max_length + 1,
                              caption_index=caption_index)
        data['targets'] = tokenized['input_ids'][:, 1:]
        if 'negative_caption_text' in data.keys():
            negative_tokenized = tokenizer(data['negative_caption_text'], return_tensors='pt',
                                           add_special_tokens=add_special_tokens,
                                           padding='max_length', max_length=max_length + 1,
                                           caption_index=caption_index)
            data['negative_targets'] = negative_tokenized['input_ids'][:, 1:]
            data['negative_inputs'] = negative_tokenized['input_ids'][:, :-1]
        if input_id_all_not_empty:
            data['inputs'] = tokenized['input_ids'][:, :-1]
        elif input_id_as_empty:
            data['inputs'] = torch.full(data['targets'].shape, tokenizer.mask_token_id)
            for number in range(max_length):
                unused_token = '[unused{}]'.format(number)
                token_id = tokenizer.convert_tokens_to_ids(unused_token)
                data['inputs'][:, number] = token_id
        else:
            data['inputs'] = tokenized['input_ids'][:, :-1]
        return data

    return build_collate_fn


def get_dataloader(dataset, config, tokenizer, is_train=False, multisos=False):
    return DataLoader(dataset, batch_size=config.training.batch_size,
                      collate_fn=collate_fn_with_tokenizer(tokenizer,
                                                           config.decoder.max_length,
                                                           config.bert.input_id_as_empty,
                                                           config.bert.input_id_all_not_empty,
                                                           multisos=multisos),
                      shuffle=is_train,
                      # num_workers=0)
                      num_workers=config.training.batch_size // 4)


if __name__ == '__main__':
    config = get_config('config/w2v.yml')
    dataset = ClothoDataset('data/clotho_captions_valid.txt', config)
    sample = dataset.__getitem__(0)
