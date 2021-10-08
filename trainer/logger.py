from torch.utils.tensorboard import SummaryWriter
import os
import time

from evaluation.eval_model import generate_caption_file


class Logger:
    def __init__(self, config, experiment_name=None):
        if experiment_name is None:
            experiment_name = config.experiment.name
        self.summary_writer = SummaryWriter(comment="-{}".format(experiment_name))
        self.experiment_name = experiment_name
        self.summary_writer.add_text('config', str(config.toDict()))
        self.best_result = {}

    def add_train_loss(self, loss, steps):
        self.summary_writer.add_scalar('train_loss', loss, steps)

    def add_train_grad(self, grads, steps):
        self.summary_writer.add_scalar('train_grad', grads, steps)

    def add_metrics(self, split, metrics, steps):
        for key in metrics.keys():
            split_key = '{}_{}'.format(split, key)
            self.summary_writer.add_scalar(split_key, metrics[key]['score'], steps)
            if split_key not in self.best_result.keys():
                self.best_result[split_key] = metrics[key]['score']
            elif split_key in self.best_result.keys() and self.best_result[split_key] < metrics[key]['score']:
                self.best_result[split_key] = metrics[key]['score']
            self.summary_writer.add_scalar("best_{}".format(split_key), self.best_result[split_key], steps)

    def format_metrics(self, split, metrics):
        metric_string = ""
        for key in metrics.keys():
            split_key = '{}_{}'.format(split, key)
            metric_string += "{}: {}\n".format(split_key, metrics[key]['score'])
        for key in list(filter(lambda x: x.startswith(split), self.best_result.keys())):
            metric_string += "BEST {}: {}\n".format(key, self.best_result[key])
        return metric_string

    def generate_captions(self, split, ground_truth, predicted):
        filename = "generated_captions/{}/{}.txt".format(self.experiment_name, split,
                                                         time.strftime("%Y-%m-%d %H.%M.%S"))
        os.makedirs("generated_captions/{}/".format(self.experiment_name, split), exist_ok=True)
        generate_caption_file(filename, ground_truth, predicted)
