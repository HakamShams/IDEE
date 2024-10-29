# ------------------------------------------------------------------
"""
Utility functions

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------

import torch
import numpy as np
import random
import os
import datetime
import logging
from functools import reduce

from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.cosine_lr import CosineLRScheduler

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

# ------------------------------------------------------------------

def log_string(logger, str):
    logger.info(str)
    print(str)


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dir(dir):
    os.makedirs(dir, exist_ok=True)


def get_logger(config):
    # Set Logger and create Directories

    if config.name is None or len(config.name) == 0:
        config.name = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    dir_log = os.path.join(config.dir_log, config.name)
    make_dir(dir_log)

    if config.phase == 'train':
        checkpoints_dir = os.path.join(dir_log, 'model_checkpoints/')
        make_dir(checkpoints_dir)

    logger = logging.getLogger("Trainer")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/log_file.txt' % dir_log)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_optimizer_groups(model, config):
    # Based on https://github.com/karpathy/minGPT/commit/bbbdac74fa9b2e55574d70056163ffbae42310c1

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()

    blacklist_weight_modules = (torch.nn.LayerNorm,
                                torch.nn.Embedding,
                                torch.nn.BatchNorm2d,
                                torch.nn.BatchNorm3d,
                                torch.nn.BatchNorm1d,
                                torch.nn.GroupNorm,
                                torch.nn.InstanceNorm1d,
                                torch.nn.InstanceNorm2d,
                                torch.nn.InstanceNorm3d)

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name##
            mm = reduce(getattr, fpn.split(sep='.')[:-1], model)
            if pn.endswith('bias') or pn.endswith('relative_position_bias_table'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(mm, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            else:
                decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay

    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(
        param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params),)

    # create the pytorch optimizer object
    optim_groups = [

        {"params": [param_dict[pn] for pn in sorted(list(decay))],
         'lr': config.en_lr, "weight_decay": config.en_weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))],
         'lr': config.en_lr, "weight_decay": 0.0},

    ]

    return optim_groups


def get_optimizer(optim_groups, config):
    optim = config.optimizer

    if optim == 'Adam':
        optimizer = torch.optim.Adam(optim_groups, betas=(config.beta1, config.beta2))
    elif optim == 'AdamW':
        optimizer = torch.optim.AdamW(optim_groups, betas=(config.beta1, config.beta2))
    # elif optim == 'SGD':
    #    optimizer = torch.optim.SGD(optim_groups, config.en_lr)
    # elif optim == 'RMSprop':
    #    optimizer = torch.optim.RMSprop(optim_groups, lr=config.en_lr)
    else:
        raise ValueError('Unexpected optimizer {} supported optimizers are Adam and AdamW'.format(config.optimizer))

    return optimizer


def get_learning_scheduler(optimizer, config):
    lr_scheduler = config.lr_scheduler

    if lr_scheduler == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=config.lr_decay_step,
            decay_rate=config.lr_decay_rate,
            warmup_lr_init=config.lr_warmup,
            warmup_t=config.lr_warmup_epochs,
            t_in_epochs=True,
        )

    elif lr_scheduler == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=config.n_epochs,
            cycle_mul=1.,
            lr_min=config.lr_min,
            warmup_lr_init=config.lr_warmup,
            warmup_t=config.lr_warmup_epochs,
            cycle_limit=1,
            t_in_epochs=False,
            warmup_prefix=False
        )

    else:
        raise ValueError('Unexpected optimizer {}, supported scheduler is step or cosine'.format(config.optimizer))

    return lr_scheduler


class evaluator():
    def __init__(self, logger, mode, config):

        self.classes = ['normal', 'drought']
        self.n_classes = len(self.classes)

        self.mode = mode
        self.config = config
        self.logger = logger

        self.correct_all = 0
        self.seen_all = 0

        self.F1 = 0

        self.weights_label = np.zeros(self.n_classes)
        self.seen_label_all = [0 for _ in range(self.n_classes)]
        self.correct_label_all = [0 for _ in range(self.n_classes)]
        self.iou_de_label_all = [0 for _ in range(self.n_classes)]
        self.predicted_label_all = [0 for _ in range(self.n_classes)]

    def get_results(self, mean_loss, best_loss):

        weights_label = self.weights_label.astype(np.float32) / np.sum(self.weights_label.astype(np.float32))
        self.accuracy_all = self.correct_all / float(self.seen_all)

        message = '-----------------   %s   -----------------\n' % self.mode

        accuracy = [0 for _ in range(self.n_classes)]
        precision = [0 for _ in range(self.n_classes)]
        F1 = [0 for _ in range(self.n_classes)]
        iou = [0 for _ in range(self.n_classes)]

        for label in range(self.n_classes):
            precision[label] = self.correct_label_all[label] / float(self.predicted_label_all[label])
            accuracy[label] = self.correct_label_all[label] / (np.array(self.seen_label_all[label], dtype=float) + 1e-6)
            F1[label] = 2 * precision[label] * accuracy[label] / (accuracy[label] + precision[label])
            iou[label] = self.correct_label_all[label] / float(self.iou_de_label_all[label])

        self.F1 = F1
        self.iou = iou

        for label in range(self.n_classes):
            message += 'class %s weight: %.4f, precision: %.4f, accuracy: %.4f, F1: %.4f IoU: %.4f \n' % (
                self.classes[label] + ' ' * (14 - len(self.classes[label])), weights_label[label],
                precision[label],
                accuracy[label],
                F1[label],
                iou[label])

        message += '\n%s accuracy      : %.4f' % (self.mode, self.accuracy_all)
        message += '\n%s mean accuracy : %.4f' % (self.mode, np.nanmean(accuracy))
        message += '\n%s mean IoU      : %.4f' % (self.mode, np.nanmean(iou))
        message += '\n%s mean F1       : %.4f' % (self.mode, np.nanmean(F1))
        message += '\n%s mean loss     : %.4f' % (self.mode, mean_loss)
        message += '\n%s best mean loss: %.4f\n' % (self.mode, best_loss)

        log_string(self.logger, message)

    def reset(self):

        self.correct_all = 0
        self.seen_all = 0
        self.F1 = 0

        self.weights_label = np.zeros(self.n_classes)
        self.seen_label_all = [0 for _ in range(self.n_classes)]
        self.correct_label_all = [0 for _ in range(self.n_classes)]
        self.iou_de_label_all = [0 for _ in range(self.n_classes)]
        self.predicted_label_all = [0 for _ in range(self.n_classes)]

    def __call__(self, VHI_p_c, VHI_c, mask):

        VHI_c = VHI_c.flatten()
        VHI_p_c = VHI_p_c.flatten()
        mask = mask.flatten()
        index = (mask > 0)
        VHI_p_c = VHI_p_c[index]
        VHI_c = VHI_c[index]

        correct = np.sum(VHI_p_c == VHI_c)
        self.correct_all += correct
        self.seen_all += len(VHI_c)

        weights, _ = np.histogram(VHI_c, range(self.n_classes + 1))
        self.weights_label += weights

        for label in range(self.n_classes):
            self.correct_label_all[label] += np.sum((VHI_p_c == label) & (VHI_c == label))
            self.seen_label_all[label] += np.sum((VHI_c == label))
            self.iou_de_label_all[label] += np.sum(((VHI_p_c == label) | (VHI_c == label)))
            self.predicted_label_all[label] += np.sum(VHI_p_c == label)


class evaluator_synthetic():
    def __init__(self, logger, mode):

        self.classes = [u' Δt0']

        self.n_classes = len(self.classes)

        self.mode = mode
        self.logger = logger

        self.correct_all = 0
        self.seen_all = 0

        self.accuracy = [0 for _ in range(self.n_classes)]
        self.precision = [0 for _ in range(self.n_classes)]
        self.F1 = [0 for _ in range(self.n_classes)]
        self.iou = [0 for _ in range(self.n_classes)]

        self.weights = np.zeros(self.n_classes)
        self.seen = [0 for _ in range(self.n_classes)]
        self.correct = [0 for _ in range(self.n_classes)]
        self.iou_de = [0 for _ in range(self.n_classes)]
        self.predicted = [0 for _ in range(self.n_classes)]

    def get_results(self, mean_loss, best_loss):

        message = '-----------------   %s   -----------------\n' % self.mode

        for label in range(self.n_classes):
            self.precision[label] = self.correct[label] / float(self.predicted[label])
            self.accuracy[label] = self.correct[label] / (
                        np.array(self.seen[label], dtype=float) + 1e-6)
            self.F1[label] = 2 * self.precision[label] * self.accuracy[label] / (
                        self.accuracy[label] + self.precision[label])
            self.iou[label] = self.correct[label] / float(self.iou_de[label])

        for label in range(self.n_classes):
            message += 'class %s weight: %.4f, precision: %.4f, accuracy: %.4f, F1: %.4f IoU: %.4f \n' % (
                self.classes[label] + ' ' * (14 - len(self.classes[label])),
                self.seen[label] / (self.seen_all / self.n_classes),
                self.precision[label],
                self.accuracy[label],
                self.F1[label],
                self.iou[label])

        # message += '\n%s accuracy      : %.4f' % (self.mode, self.accuracy_all)
        message += '\n%s mean accuracy : %.4f' % (self.mode, np.nanmean(self.accuracy))
        message += '\n%s mean IoU      : %.4f' % (self.mode, np.nanmean(self.iou))
        message += '\n%s mean F1       : %.4f' % (self.mode, np.nanmean(self.F1))
        message += '\n%s mean loss     : %.4f' % (self.mode, mean_loss)
        message += '\n%s best mean loss: %.4f\n' % (self.mode, best_loss)

        log_string(self.logger, message)

    def reset(self):

        self.correct_all = 0
        self.seen_all = 0

        self.accuracy = [0 for _ in range(self.n_classes)]
        self.precision = [0 for _ in range(self.n_classes)]
        self.F1 = [0 for _ in range(self.n_classes)]
        self.iou = [0 for _ in range(self.n_classes)]

        self.weights = np.zeros(self.n_classes)
        self.seen = [0 for _ in range(self.n_classes)]
        self.correct = [0 for _ in range(self.n_classes)]
        self.iou_de = [0 for _ in range(self.n_classes)]
        self.predicted = [0 for _ in range(self.n_classes)]

    def __call__(self, pred_c, gt):

        self.seen_all += len(gt.flatten())

        for label in range(self.n_classes):
            self.correct[label] += np.sum((pred_c[:, label, ...] == 1) & (gt[:, label, ...] == 1))
            self.seen[label] += np.sum((gt[:, label, ...] == 1))
            self.iou_de[label] += np.sum(((pred_c[:, label, ...] == 1) | (gt[:, label, ...] == 1)))
            self.predicted[label] += np.sum(pred_c[:, label, ...] == 1)


class evaluator_anomaly_synthetic():
    def __init__(self, logger, mode, config):

        self.classes = config.variables
        self.n_classes = len(self.classes)

        self.mode = mode
        self.logger = logger

        self.correct_all = 0
        self.seen_all = 0

        self.accuracy_pos = [0 for _ in range(self.n_classes)]
        self.precision_pos = [0 for _ in range(self.n_classes)]
        self.F1_pos = [0 for _ in range(self.n_classes)]
        self.iou_pos = [0 for _ in range(self.n_classes)]

        self.accuracy_neg = [0 for _ in range(self.n_classes)]
        self.precision_neg = [0 for _ in range(self.n_classes)]
        self.F1_neg = [0 for _ in range(self.n_classes)]
        self.iou_neg = [0 for _ in range(self.n_classes)]

        self.accuracy_p_all = 0
        self.precision_all = 0
        self.F1_all = 0
        self.iou_all = 0

        self.seen_pos = [0 for _ in range(self.n_classes)]
        self.correct_pos = [0 for _ in range(self.n_classes)]
        self.iou_de_pos = [0 for _ in range(self.n_classes)]
        self.predicted_pos = [0 for _ in range(self.n_classes)]

        self.seen_neg = [0 for _ in range(self.n_classes)]
        self.correct_neg = [0 for _ in range(self.n_classes)]
        self.iou_de_neg = [0 for _ in range(self.n_classes)]
        self.predicted_neg = [0 for _ in range(self.n_classes)]

        self.seen_p_all = 0
        self.correct_p_all = 0
        self.iou_de_all = 0
        self.predicted_all = 0

        self.FP = [0 for _ in range(self.n_classes)]
        self.FN = [0 for _ in range(self.n_classes)]

    def get_results(self):

        self.accuracy_all = self.correct_all / float(self.seen_all)

        message = '-----------------   %s   -----------------\n' % self.mode

        for label in range(self.n_classes):
            self.precision_pos[label] = self.correct_pos[label] / float(self.predicted_pos[label])
            self.accuracy_pos[label] = self.correct_pos[label] / (
                        np.array(self.seen_pos[label], dtype=float) + 1e-6)
            self.F1_pos[label] = 2 * self.precision_pos[label] * self.accuracy_pos[label] / (self.accuracy_pos[label] + self.precision_pos[label])
            self.iou_pos[label] = self.correct_pos[label] / float(self.iou_de_pos[label])

            self.precision_neg[label] = self.correct_neg[label] / float(self.predicted_neg[label])
            self.accuracy_neg[label] = self.correct_neg[label] / (
                        np.array(self.seen_neg[label], dtype=float) + 1e-6)
            self.F1_neg[label] = 2 * self.precision_neg[label] * self.accuracy_neg[label] / (self.accuracy_neg[label] + self.precision_neg[label])
            self.iou_neg[label] = self.correct_neg[label] / float(self.iou_de_neg[label])

        self.precision_all = self.correct_p_all / float(self.predicted_all)
        self.accuracy_p_all = self.correct_p_all / (np.array(self.seen_p_all, dtype=float) + 1e-6)
        self.F1_all = 2 * self.precision_all * self.accuracy_p_all / (self.accuracy_p_all + self.precision_all)
        self.iou_all = self.correct_p_all / float(self.iou_de_all)

        for label in range(self.n_classes):
            message += 'class %s pos   weight: %.4f, precision: %.4f, accuracy: %.4f, F1: %.4f IoU: %.4f \n' % (
                self.classes[label] + ' ' * (7 - len(self.classes[label])),
                self.seen_pos[label] / (self.seen_all / self.n_classes),
                self.precision_pos[label],
                self.accuracy_pos[label],
                self.F1_pos[label],
                self.iou_pos[label])

            message += ' ' * (13 + 7 - len(self.classes[label])) + 'neg   weight: %.4f, precision: %.4f, accuracy: %.4f, F1: %.4f IoU: %.4f \n' % (
                self.seen_neg[label] / (self.seen_all / self.n_classes),
                self.precision_neg[label],
                self.accuracy_neg[label],
                self.F1_neg[label],
                self.iou_neg[label])

        message += '\n'
        for label in range(self.n_classes):
            message += 'class %s weight: %.4f, TP: %i, FP: %i, TN: %i FN: %i, F1: %.4f, IoU: %.4f \n' % (
                self.classes[label] + ' ' * (13 - len(self.classes[label])),
                self.seen_pos[label] / (self.seen_all / self.n_classes),
                self.correct_pos[label],
                self.FP[label],
                self.correct_neg[label],
                self.FN[label],
                self.F1_pos[label],
                self.iou_pos[label])

        message += '\n'
        message += 'all var             weight: %.4f, precision: %.4f, accuracy: %.4f, F1: %.4f IoU: %.4f \n' % (
            self.seen_p_all / self.seen_all,
            self.precision_all,
            self.accuracy_p_all,
            self.F1_all,
            self.iou_all)

        message += '\n%s accuracy               : %.4f' % (self.mode, self.accuracy_all)
        message += '\n%s mean accuracy positive : %.4f' % (self.mode, np.nanmean(self.accuracy_pos))
        message += '\n%s mean IoU positive      : %.4f' % (self.mode, np.nanmean(self.iou_pos))
        message += '\n%s mean F1 positive       : %.4f' % (self.mode, np.nanmean(self.F1_pos))

        log_string(self.logger, message)

    def reset(self):

        # self.__init__() bad practice
        self.correct_all = 0
        self.seen_all = 0

        self.accuracy_pos = [0 for _ in range(self.n_classes)]
        self.precision_pos = [0 for _ in range(self.n_classes)]
        self.F1_pos = [0 for _ in range(self.n_classes)]
        self.iou_pos = [0 for _ in range(self.n_classes)]

        self.accuracy_neg = [0 for _ in range(self.n_classes)]
        self.precision_neg = [0 for _ in range(self.n_classes)]
        self.F1_neg = [0 for _ in range(self.n_classes)]
        self.iou_neg = [0 for _ in range(self.n_classes)]

        self.accuracy_p_all = 0
        self.precision_all = 0
        self.F1_all = 0
        self.iou_all = 0

        self.seen_pos = [0 for _ in range(self.n_classes)]
        self.correct_pos = [0 for _ in range(self.n_classes)]
        self.iou_de_pos = [0 for _ in range(self.n_classes)]
        self.predicted_pos = [0 for _ in range(self.n_classes)]

        self.seen_neg = [0 for _ in range(self.n_classes)]
        self.correct_neg = [0 for _ in range(self.n_classes)]
        self.iou_de_neg = [0 for _ in range(self.n_classes)]
        self.predicted_neg = [0 for _ in range(self.n_classes)]

        self.seen_p_all = 0
        self.correct_p_all = 0
        self.iou_de_all = 0
        self.predicted_all = 0

        self.FP = [0 for _ in range(self.n_classes)]
        self.FN = [0 for _ in range(self.n_classes)]

    def __call__(self, VHI_p_c, VHI_c):

        correct = np.sum(VHI_p_c == VHI_c)

        self.correct_all += correct
        self.seen_all += VHI_c.size

        for label in range(self.n_classes):

            self.correct_pos[label] += np.sum((VHI_p_c[:, label, ...] == 1) & (VHI_c[:, label, ...] == 1))
            self.seen_pos[label] += np.sum((VHI_c[:, label, ...] == 1))
            self.iou_de_pos[label] += np.sum(((VHI_p_c[:, label, ...] == 1) | (VHI_c[:, label, ...] == 1)))
            self.predicted_pos[label] += np.sum(VHI_p_c[:, label, ...] == 1)

            self.correct_neg[label] += np.sum((VHI_p_c[:, label, ...] == 0) & (VHI_c[:, label, ...] == 0))
            self.seen_neg[label] += np.sum((VHI_c[:, label, ...] == 0))
            self.iou_de_neg[label] += np.sum(((VHI_p_c[:, label, ...] == 0) | (VHI_c[:, label, ...] == 0)))
            self.predicted_neg[label] += np.sum(VHI_p_c[:, label, ...] == 0)

            self.FP[label] += np.sum((VHI_p_c[:, label, ...] == 1) & (VHI_c[:, label, ...] == 0))
            self.FN[label] += np.sum((VHI_p_c[:, label, ...] == 0) & (VHI_c[:, label, ...] == 1))

        self.correct_p_all += np.sum((VHI_p_c == 1) & (VHI_c == 1))
        self.seen_p_all += np.sum((VHI_c == 1))
        self.iou_de_all += np.sum(((VHI_p_c == 1) | (VHI_c == 1)))
        self.predicted_all += np.sum(VHI_p_c == 1)


class anomaly_collector:
    def __init__(self, anomaly: np.array, timestep: np.array, config):

        self.anomaly = np.zeros(anomaly.shape)
        self.timestep = timestep
        self.delta_t = config.delta_t
        self.anomaly_counter = np.zeros(anomaly.shape)

    def reset(self):
        self.anomaly = np.zeros_like(self.anomaly)
        self.anomaly_counter = np.zeros_like(self.anomaly_counter)

    def majority_vote(self):

        self.anomaly = self.anomaly / self.anomaly_counter
        self.anomaly[self.anomaly >= 0.5] = 1
        self.anomaly[self.anomaly < 0.5] = 0

    def __call__(self, anomaly, timestep):

        for n in range(len(timestep)):

            idx = np.argwhere(timestep[n] == self.timestep).item()

            self.anomaly[:, idx - self.delta_t + 1: idx + 1, ...] += np.flip(anomaly[n], axis=1)
            self.anomaly_counter[:, idx - self.delta_t + 1: idx + 1, ...] += 1


def min_max_scale(array, min_array, max_array, min_new=-1., max_new=1.):
    array = ((max_new - min_new) * (array - min_array) / (max_array - min_array)) + min_new
    return array


def save_model(model, optimizer, epoch, mean_loss_train, mean_loss_val, logger, config, metric):
    dir_log = os.path.join(config.dir_log, config.name)
    checkpoints_dir = os.path.join(dir_log, 'model_checkpoints/')
    if metric == 'loss':
        path = os.path.join(checkpoints_dir, 'best_loss_model.pth')
        log_string(logger, 'saving model to %s' % path)

    elif metric == 'F1':
        path = os.path.join(checkpoints_dir, 'best_F1_model.pth')
        log_string(logger, 'saving model to %s' % path)

    elif metric == 'train':
        path = os.path.join(checkpoints_dir, 'best_train_model.pth')

    state = {
        'epoch': epoch,
        'mean_loss_train': mean_loss_train,
        'mean_loss_validation': mean_loss_val,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    torch.save(state, path)


def get_colors(inp, colormap, vmin=None, vmax=None):
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))

def generate_images(pred, pred_c, target, mask_cold, mask_sea, mask_no_vegetation, mask_prudence):
    norm = plt.Normalize(0, 1)

    im_pred = plt.cm.cividis(norm(pred))[:, :, :, :-1]

    pred_c[mask_prudence == 0] = 0
    pred_c[mask_sea == 1] = 2
    pred_c[mask_no_vegetation == 1] = 3
    pred_c[mask_cold == 1] = 4
    target[mask_prudence == 0] = 0
    target[mask_sea == 1] = 2
    target[mask_no_vegetation == 1] = 3
    target[mask_cold == 1] = 4

    color_dict = {0: 'wheat',
                  1: 'darkred',
                  2: 'darkcyan',
                  3: 'lightgrey',
                  4: 'white'
                  }

    cm = ListedColormap(color_dict.values())
    im_pred_c = cm(pred_c.astype(np.uint8))[:, :, :, :-1]
    im_target = cm(target.astype(np.uint8))[:, :, :, :-1]

    return im_pred, im_pred_c, im_target


def generate_anomaly(anomaly):
    N, V, T, H, W = anomaly.shape

    #anomaly[anomaly >= 0.5] = 1
    #anomaly[anomaly < 0.5] = 0

    im_anomaly = np.zeros((N, V, H, W * T, 3))

    color_dict = {0: 'wheat',
                  1: 'darkred',
                  }

    cm = ListedColormap(color_dict.values())

    for v in range(V):
        # N, T, H, W
        im_t = anomaly[:, v, :, :, :]
        im_tmp = np.zeros((N, H, W * T))
        for t in range(T):
            im_tmp[:, :, W * t:W * (t + 1)] = im_t[:, t, :, :]
        im_tmp_c = cm(im_tmp.astype(np.uint8))[:, :, :, :-1]
        im_anomaly[:, v, :, :, :] = im_tmp_c

    return im_anomaly


def generate_images_synthetic(pred, pred_c, target):
    norm = plt.Normalize(0, 1)
    im_pred = plt.cm.cividis(norm(pred))[:, :, :, :-1]

    color_dict = {0: 'wheat',
                  1: 'darkred',
                  }

    cm = ListedColormap(color_dict.values())
    im_pred_c = cm(pred_c.astype(np.uint8))[:, :, :, :-1]
    im_target = cm(target.astype(np.uint8))[:, :, :, :-1]

    return im_pred, im_pred_c, im_target


if __name__ == '__main__':

    generate_anomaly(np.random.randn(1, 12, 8, 128, 208))

