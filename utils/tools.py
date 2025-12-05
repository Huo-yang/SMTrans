import numpy as np
import torch
import os
import json
import sys
import math
import re


class CheckpointSaver:
    def __init__(self, model_name, verbose=False, delta=0, times_now=None):
        self.model_name = model_name
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_acc = None
        self.val_loss_min = np.inf
        self.vali_acc_max = np.inf
        self.delta = delta
        self.save_checkpoint_path = None
        self.times_now = times_now

    def __call__(self, logger, val_loss, results, path, vali_acc=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_acc = vali_acc
            self.save_checkpoint(logger, val_loss, vali_acc, results, path)
            save_flag = True
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f'Loss haven\'t decreased count: {self.counter}')
            save_flag = False
        else:
            self.best_score = score
            self.best_acc = vali_acc
            self.save_checkpoint(logger, val_loss, vali_acc, results, path)
            self.counter = 0
            save_flag = True
        return self.save_checkpoint_path, save_flag

    def save_checkpoint(self, logger, val_loss, vali_acc, results, path):
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
            # print(f', and validation acc increased ({self.vali_acc_max:.6f} --> {vali_acc:.6f}).')
        if self.save_checkpoint_path is not None:
            os.remove(self.save_checkpoint_path)
        if vali_acc is None:
            self.save_checkpoint_path = os.path.join(path, '{}_{}_L[{:.4f}].pth.tar'.format(
                self.times_now if self.times_now is not None else 0,
                self.model_name, val_loss))
        else:
            self.save_checkpoint_path = os.path.join(path, '{}_{}_A[{:.4f}]_L[{:.4f}].pth.tar'.format(
                self.times_now if self.times_now is not None else 0,
                self.model_name, vali_acc, val_loss))
        logger.info(f"saving model to {self.save_checkpoint_path}")
        torch.save(results, self.save_checkpoint_path)
        self.val_loss_min = val_loss
        self.vali_acc_max = vali_acc


class LearningRateAdjuster():
    def __init__(self, initial_lr: float, patience: int, lr_decay_rate: float = 0.5, type: str = "type1"):
        self.lr = initial_lr
        self.min_lr = 5e-5
        self.patience = patience
        self.lr_decay_rate = lr_decay_rate
        assert 0 < self.lr_decay_rate <= 1, "Learning rate decay rate should be between 0 and 1."
        self.type = type

    def _update_lr(self, logger, optimizer):
        if self.lr > self.min_lr:
            self.lr *= self.lr_decay_rate
        else:
            self.lr = self.min_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr
        logger.info(f'Updating learning rate to {self.lr}')

    def rate_decay_with_patience(self, logger, optimizer, patience_count):
        if self.type == "type1":
            if patience_count / self.patience > 0.5 and self.lr != self.min_lr:
                self._update_lr(logger, optimizer)
        elif self.type == "type2":
            if patience_count % 5 == 0 and patience_count != 0 and self.lr != self.min_lr:
                self._update_lr(logger, optimizer)
        elif self.type == "type3":
            if (patience_count - 5) % 3 == 0 and patience_count >= 5 and self.lr != self.min_lr:
                self._update_lr(logger, optimizer)


class EarlyStopping(CheckpointSaver):
    def __init__(self, model_name, patience=7, verbose=False, delta=0, times_now=None):
        super().__init__(model_name, verbose, delta, times_now)
        self.patience = patience
        self.early_stop = False

    def __call__(self, logger, val_loss, results, path, vali_acc=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_acc = vali_acc
            self.save_checkpoint(logger, val_loss, vali_acc, results, path)
            save_flag = True
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            save_flag = False
        else:
            self.best_score = score
            self.best_acc = vali_acc
            self.save_checkpoint(logger, val_loss, vali_acc, results, path)
            self.counter = 0
            save_flag = True
        return self.save_checkpoint_path, save_flag


class LossMeter:
    def __init__(self, start_epoch):
        self.start_epoch = start_epoch
        self.end_epoch = start_epoch
        self.all_loss = []
        self.current_loss = []

    def __call__(self, loss):
        self.current_loss.append(loss)

    def epoch_step(self):
        self.end_epoch += 1
        self.all_loss.append(self.current_loss)
        self.current_loss = []

    def avg_epoch_loss(self):
        return np.average(self.current_loss)


class AccMeter:
    def __init__(self, start_epoch):
        self.start_epoch = start_epoch
        self.end_epoch = start_epoch
        self.acc = []

    def __call__(self, acc):
        self.acc.append(acc)

    def epoch_step(self):
        self.end_epoch += 1

    def epoch_acc(self):
        return self.acc[-1]


class StandardScaler():
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean


def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)
    return args


def save_model_structure_in_txt(path, model):
    with open(os.path.join(path, 'model_structure.txt'), 'w') as f:
        sys.stdout = f
        print(model)
        sys.stdout = sys.__stdout__


def parse_string_to_list(str_for_split: str, flag: str = "float"):
    str_no_space = str_for_split.replace(' ', '')
    str_split = str_no_space.split(',')
    value_list = []
    
    for item in str_split:
        # 处理 "pi" 常量
        item = item.replace('pi', f'*{math.pi}') if 'pi' in item and '*' not in item else item
        try:
            value = eval(item, {"__builtins__": {}}, {})
            if flag == "int":
                value = int(value)
            elif flag == "float":
                value = float(value)
            else:
                raise ValueError("flag must be 'int' or 'float'")
            value_list.append(value)
        except Exception as e:
            raise ValueError(f"Invalid expression: '{item}', error: {e}")
    return value_list



def snr_string_to_list(snr_string: str):
    matches = re.findall(r'N?\d+', snr_string)
    matches = [int(m.replace('N', '-')) for m in matches]
    assert len(matches) == 2
    SNRs = list(range(matches[0], matches[1] + 2, 2))
    return SNRs


def describe_augmentations(transform):
    """
    获取一个transform序列（如 Compose）中的每个增强名称和参数
    """
    descriptions = []
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            cls_name = type(t).__name__
            if hasattr(t, '__dict__') and t.__dict__:  # 有参数
                params = ', '.join(f"{k}={repr(v)}" for k, v in t.__dict__.items())
                descriptions.append(f"{cls_name}({params})")
            else:  # 无参数
                descriptions.append(f"{cls_name}()")
    else:
        descriptions.append(type(transform).__name__)
    return descriptions
