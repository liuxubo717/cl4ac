from gensim.models.word2vec import Word2Vec
from re import sub
from pathlib import Path
import csv
import torch


def clean_sentence(sentence):
    the_sentence = sentence.lower()

    # remove fogotten space before punctuation and double space
    the_sentence = sub(r'\s([,.!?;:"](?:\s|$))', r'\1', the_sentence).replace('  ', ' ')

    the_sentence = sub('[,.!?;:\"]', "", the_sentence)

    return the_sentence


def main(multisos=False):
    caption_fields = ['caption_{}'.format(i) for i in range(1, 6)]

    development_csv_file = 'data/clotho_captions_train.txt'
    print('Reading csv file...')
    # read csv_files
    with open(development_csv_file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        development_csv = [csv_line for csv_line in csv_reader]
    print('Reading all captions')
    # read all captions into list
    for item in development_csv:
        captions = [clean_sentence(item.get(caption_field)) for caption_field in caption_fields]
        if multisos:
            captions = ['<sos{}> {} <eos>'.format(caption_index, caption) for
                        caption_index, caption in enumerate(captions)]
        else:
            captions = ['<sos> {} <eos>'.format(caption) for caption in captions]

        [item.update({caption_field: caption})
         for caption_field, caption in zip(caption_fields, captions)]
    # split all captions into words
    dev_captions = [item.get(caption_field).split()
                    for item in development_csv
                    for caption_field in caption_fields]
    print('Start training the model')
    # train the model
    model = Word2Vec(dev_captions, size=128, min_count=1, iter=1000)

    save_dir = Path('data/pretrained_models/word2vec')
    save_dir.mkdir(parents=True, exist_ok=True)

    print('Training finished\nSaving model in {}'.format(str(save_dir)))
    if multisos:
        model.save(str(save_dir) + '/w2v-multi-sos.model')
    else:
        model.save(str(save_dir) + '/w2v.model')


if __name__ == '__main__':
    main()
