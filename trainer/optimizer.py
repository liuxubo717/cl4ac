import torch


def get_optimizer(config, model):
    if config.w2v.enable:
        optimizer_args = [
            {'params': model.parameters(), 'lr': config.optimizer.lr, 'weight_decay': 1e-6}
        ]
    else:
        bert_keywords = ["bert_model.{}".format(k[0]) for k in model.bert_model.named_parameters()]
        bert_params = list(filter(lambda kv: kv[0] in bert_keywords, model.named_parameters()))
        other_params = list(filter(lambda kv: kv[0] not in bert_keywords, model.named_parameters()))
        optimizer_args = [
            {'params': [p[1] for p in other_params], 'lr': config.optimizer.lr, 'weight_decay': 1e-6},
            {'params': [p[1] for p in bert_params], 'lr': config.optimizer.bert_lr, 'weight_decay': 1e-6}
        ]
    if config.optimizer.optimizer == 'adam':
        optimizer = torch.optim.Adam(optimizer_args)
    else:
        raise ValueError("Use a valid optimizer!")
    return optimizer