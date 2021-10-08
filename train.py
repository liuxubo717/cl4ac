import argparse
from config_loader.config import get_config
from trainer.trainer import train
from utils.parser_helper import str2bool
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--freeze_bert', type=str2bool, default=None)
parser.add_argument('--freeze_cnn', type=str2bool, default=None)
parser.add_argument('--batch', type=int, default=None)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--experiment_name', type=str, default=None)
parser.add_argument('--lr', type=float, default=None)

args = parser.parse_args()
config_path = args.config
freeze_bert = args.freeze_bert
freeze_cnn = args.freeze_cnn
batch_size = args.batch
lr = args.lr
experiment_name = args.experiment_name
device = args.device
config = get_config(config_path)
if batch_size is not None:
    config.training.batch_size = args.batch
if freeze_bert is not None:
    config.bert.freeze_bert = freeze_bert
if freeze_cnn is not None:
    config.encoder.freeze_cnn = freeze_cnn
if lr is not None:
    config.optimizer.lr = lr
if experiment_name is not None:
    config.experiment.name = experiment_name
config.experiment.name = "{}-LR{}".format(config.experiment.name, config.optimizer.lr)

train(config, device)