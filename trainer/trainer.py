import torch
from tqdm import tqdm
import numpy as np
import random
from warmup_scheduler import GradualWarmupScheduler
from datetime import datetime
from bert_tools.custom_tokenizer import CUSTOM_TOKENIZER
from data_loader.clotho_dataset import ClothoDataset, get_dataloader
from evaluation.eval_model import eval_model
from model.TransModel import TransformerModel
from transformers import AutoTokenizer
from trainer.get_gradient import get_total_grad_norm
from trainer.logger import Logger
from trainer.loss import get_loss, calculating_weight
from trainer.optimizer import get_optimizer
from utils.save_load_model import save_model
from w2v_tools.w2v_model import Word2Vec
import os


def train(config, device):
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d+%H-%M-%S")
    setup_seed(config.training.seed)
    if config.w2v.enable:
        tokenizer = Word2Vec(w2v_model_path=config.w2v.w2v_path, multisos=config.multisos.enable)
    elif config.bert.use_custom_tokenizer:
        tokenizer = CUSTOM_TOKENIZER
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.bert.bert_path)
    train_dataset = ClothoDataset('data/clotho_captions_train.txt', config,
                                  tokenizer=tokenizer)
    valid_dataset = ClothoDataset('data/clotho_captions_valid.txt', config,
                                  tokenizer=tokenizer, is_train=False)
    test_dataset = ClothoDataset('data/clotho_captions_test.txt', config,
                                 tokenizer=tokenizer, is_train=False)
    train_loader = get_dataloader(train_dataset, config, tokenizer, is_train=True, multisos=config.multisos.enable)
    valid_loader = get_dataloader(valid_dataset, config, tokenizer, is_train=False, multisos=config.multisos.enable)
    test_loader = get_dataloader(test_dataset, config, tokenizer, is_train=False, multisos=config.multisos.enable)

    model = TransformerModel(config).to(device)
    criteria = get_loss(config)
    optimizer = get_optimizer(config, model)
    if config.training.activate_weight_on_loss.enable:
        addition_weight_ratio = train_dataset.get_word_frequency(tokenizer)
        weight = calculating_weight(tokenizer, addition_weight_ratio,
                                    reduce_punc=config.training.activate_weight_on_loss.reduce_punc_weight,
                                    reduce_stopwords=config.training.activate_weight_on_loss.reduce_punc_weight)
        criteria.weight = torch.tensor(weight).to(device)
    # Must be set after weight calculation, since [PAD] weight will be set as 0
    if config.bert.use_sep_as_pad:
        tokenizer.pad_token = tokenizer.sep_token
    logger = Logger(config)
    steps = 0
    scheduler_warmup = None
    if config.optimizer.warm_up.enable:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)

    for epoch in range(config.training.epoch):
        epoch_loss = []
        pbar = tqdm(train_loader, total=train_loader.__len__(),
                    position=0, leave=True, ascii=True, desc="Epoch {}".format(epoch))
        if config.optimizer.warm_up.enable:
            scheduler_warmup.step(epoch + 1)
        for data in pbar:
            model.train()
            optimizer.zero_grad()
            attention_mask = None if 'attention_mask' not in data.keys() else data['attention_mask'].to(device)
            max_non_pad_indexes = None
            max_neg_non_pad_indexes = None
            if config.auxiliary_task.use_last_hidden:
                non_pad_indexes = (data['inputs'] != tokenizer.pad_token_id).nonzero()
                max_non_pad_indexes = torch.zeros((data['inputs'].shape[0]))
                for non_pad_index in non_pad_indexes.detach().cpu().numpy().tolist():
                    max_non_pad_indexes[non_pad_index[0]] = non_pad_index[1]
                neg_non_pad_indexes = (data['negative_inputs'] != tokenizer.pad_token_id).nonzero()
                max_neg_non_pad_indexes = torch.zeros((data['negative_inputs'].shape[0]))
                for non_pad_index in neg_non_pad_indexes.detach().cpu().numpy().tolist():
                    max_neg_non_pad_indexes[non_pad_index[0]] = non_pad_index[1]

            y_hat = model(data['audio_embedding'].to(device), data['inputs'].to(device),
                          attention_mask=attention_mask,
                          selection_result=config.auxiliary_task.selection_loss,
                          max_non_pad_indexes=max_non_pad_indexes.to(device))
            if config.auxiliary_task.selection_loss:
                y_hat, selection_score = y_hat
                _, negative_selection_score = model(data['audio_embedding'].to(device),
                                                    data['negative_inputs'].to(device),
                                                    attention_mask=attention_mask,
                                                    selection_result=config.auxiliary_task.selection_loss,
                                                    max_non_pad_indexes=max_neg_non_pad_indexes.to(device))
                selection_labels = [1] * selection_score.shape[0] + [0] * negative_selection_score.shape[0]
                pos_neg_selection_scores = torch.cat([selection_score, negative_selection_score])
                selection_loss = criteria(pos_neg_selection_scores,
                                          torch.tensor(selection_labels).to(device).contiguous().view(-1))
            if config.bert.auto_regression and config.bert.auto_regressive_gamma < 1.0:
                losses = None
                for batch_index in range(y_hat.shape[0]):
                    y_hat_iter = y_hat[batch_index]
                    input_iter = data['inputs'][batch_index]
                    targets_iter = data['targets'][batch_index]
                    known_length = torch.nonzero(input_iter != tokenizer.mask_token_id).shape[0]
                    gamma = config.bert.auto_regressive_gamma ** known_length
                    loss_iter = criteria(y_hat_iter, targets_iter.to(device)) * gamma
                    if losses is None:
                        losses = loss_iter
                    else:
                        losses += losses
                losses /= y_hat.shape[0]
                loss = losses
            else:
                loss = criteria(y_hat.contiguous().view(-1, y_hat.shape[-1]),
                                data['targets'].to(device).contiguous().view(-1))
            if config.auxiliary_task.selection_loss:
                loss += selection_loss
            loss.backward()
            epoch_loss.append(loss.detach().cpu())
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            gradients = get_total_grad_norm(model.parameters())
            optimizer.step()
            logger.add_train_loss(loss.detach().cpu().item(), steps)
            logger.add_train_grad(gradients, steps)
            first_token_ids = torch.argmax(y_hat, dim=-1).detach().cpu().numpy().tolist()[0]
            pbar.set_postfix_str("loss: {:.4f}, gradient: {:.4f} pred: {}".format(
                loss.detach().cpu().item(), gradients,
                tokenizer.decode(first_token_ids)
            ))
            steps += 1

        valid_metrics, valid_ground_truth, valid_predicted, beam_valid_metrics, beam_valid_predicted = eval_model(model,
                                                                                                                  valid_loader,
                                                                                                                  tokenizer,
                                                                                                                  config,
                                                                                                                  device)
        test_metrics, test_ground_truth, test_predicted, beam_test_metrics, beam_test_predicted = eval_model(model,
                                                                                                             test_loader,
                                                                                                             tokenizer,
                                                                                                             config,
                                                                                                             device)
        logger.add_metrics('valid', valid_metrics, steps)
        logger.add_metrics('test', test_metrics, steps)
        logger.add_metrics('beam_valid', beam_valid_metrics, steps)
        logger.add_metrics('beam_test', beam_test_metrics, steps)
        logger.generate_captions('valid_{}_Epoch{:03d}'.format(config.experiment.name, epoch), valid_ground_truth,
                                 valid_predicted)
        logger.generate_captions('test_{}_Epoch{:03d}'.format(config.experiment.name, epoch), test_ground_truth,
                                 test_predicted)
        logger.generate_captions('valid_{}_Epoch{:03d}_beam'.format(config.experiment.name, epoch), valid_ground_truth,
                                 beam_valid_predicted)
        logger.generate_captions('test_{}_Epoch{:03d}_beam'.format(config.experiment.name, epoch), test_ground_truth,
                                 beam_test_predicted)
        print(logger.format_metrics('valid', valid_metrics))
        print(logger.format_metrics('beam_valid', beam_valid_metrics))
        print(logger.format_metrics('test', test_metrics))
        print(logger.format_metrics('beam_test', beam_test_metrics))
        # Save model every epoch
        # model_path = 'saved_models/{}-{}/'.format(config.experiment.name, current_time)
        # os.makedirs(model_path, exist_ok=True)
        # model_path += "{:04d}.pt".format(epoch)
        # save_model(model_path, model, optimizer, epoch, config, np.asarray(epoch_loss).mean())


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
