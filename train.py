import argparse
import random
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import models
from utils.general import init_experiment
from utils.tools import *
from utils.result_visualization import ResultGenerator
from datasets.get_dataset import get_datasets


def get_dataloader(args):
    train_dataset, vali_dataset, test_dataset = get_datasets(args)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False)
    vali_loader = DataLoader(
        dataset=vali_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False)
    args.logger.info(f"train sample size:{len(train_dataset)}, validation sample size:{len(vali_dataset)}, test sample size:{len(test_dataset)}.")
    return train_loader, vali_loader, test_loader


def build_model(args):
    model = getattr(models, "{}".format(args.model))(num_classes=len(args.choose_classes), sig_size=args.sig_size)
    if args.resume:
        if os.path.isfile(args.resume):
            args.logger.info(f"loading checkpoint '{args.resume}'")
            model.load_state_dict(torch.load(args.resume['state_dict']))
        else:
            raise TypeError("=> no checkpoint found at '{}'".format(args.resume))
    total_params = sum(p.numel() for p in model.parameters())
    args.logger.info(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    args.logger.info(f'{total_trainable_params:,} training parameters.')
    return model


def train(args, times_now):
    device = torch.device('cuda:{}'.format(args.gpu))
    train_loader, vali_loader, test_loader = get_dataloader(args)
    model = build_model(args).to(device)
    save_model_structure_in_txt(args.result_root_path, model)

    lr_adjuster = LearningRateAdjuster(initial_lr=args.lr, patience=args.patience, lr_decay_rate=0.5,
                                       type=args.lradj)
    cpt_saver = EarlyStopping(args.model, patience=args.patience, verbose=True, delta=0.001,
                                    times_now=times_now)
    model_optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.amp.GradScaler() if args.use_scaler and torch.cuda.is_available() else None
    criterion = nn.CrossEntropyLoss()

    if args.resume:
        if os.path.isfile(args.resume):
            args.logger.info(f"[Info]loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            model_optim.load_state_dict(checkpoint['optimizer'])
            args.logger.info(f"[Info]loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            raise ValueError("no checkpoint found at '{}'".format(args.resume))

    result_generator = ResultGenerator(path=args.result_root_path, class_names=args.choose_classes,
                                       snrs=args.choose_snrs, times_now=times_now)
    meters = {
        "train_loss": LossMeter(start_epoch=args.start_epoch),
        "train_acc": AccMeter(start_epoch=args.start_epoch),
        "vali_loss": LossMeter(start_epoch=args.start_epoch),
        "vali_acc": AccMeter(start_epoch=args.start_epoch),
    }
    best_model_path = ''
    for epoch in range(args.start_epoch, args.train_epochs):
        model.train()
        train_acc = {"current": 0, "total": 0}
        for train_iterate, (batch_x, batch_y, _) in enumerate(train_loader):
            model_optim.zero_grad()
            if type(batch_x) == list:
                batch_x = (x.to(device) for x in batch_x)
            else:
                batch_x = batch_x.to(device)
            true = batch_y.to(device)
            if scaler is not None:
                with torch.amp.autocast(device_type='cuda', enabled=True):
                    predict = model(batch_x)
                    loss = criterion(predict, true)
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                predict = model(batch_x)
                loss = criterion(predict, true)
                loss.backward()
                model_optim.step()
            meters["train_loss"](loss.item())
            predict = torch.argmax(predict, 1)
            train_acc["current"] += (predict == true).sum().item()
            train_acc["total"] += len(batch_y)
            if (train_iterate + 1) % args.print_freq == 0:
                args.logger.info(f"Epoch: [{epoch + 1}][{train_iterate + 1}/{len(train_loader)}] | loss: {loss.item():.7f}")
        meters["train_acc"](train_acc["current"] / train_acc["total"])
        vali_loss, vali_acc = validation(args, vali_loader, model, criterion, device)
        meters["vali_loss"](vali_loss)
        meters["vali_acc"](vali_acc)
        args.logger.info(f"Train Epoch: {epoch + 1} | Avg Train Loss: {meters['train_loss'].avg_epoch_loss():.4f}"
                         f" Avg Train Acc: {meters['train_acc'].epoch_acc():.4f}"
                         f" | Avg vali Loss: {meters['vali_loss'].avg_epoch_loss():.4f}"
                         f"  Avg vali Acc: {meters['vali_acc'].epoch_acc():.4f}")
        best_model_path, save_flag = cpt_saver(args.logger, vali_loss, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': model_optim.state_dict(),
        }, args.result_root_path, vali_acc)
        if cpt_saver.early_stop:
            args.logger.info("Early stopping")
            break
        for key, meter in meters.items():
            meter.epoch_step()
        lr_adjuster.rate_decay_with_patience(args.logger, model_optim, cpt_saver.counter)
        result_generator.plot_loss(meters["train_loss"].all_loss, start_epoch=meters["train_loss"].start_epoch,
                                   end_epoch=meters["train_loss"].end_epoch, validation_loss_list=meters["vali_loss"].all_loss)
        result_generator.plot_acc(meters["vali_acc"].acc, start_epoch=meters["vali_acc"].start_epoch,
                                  end_epoch=meters["vali_acc"].end_epoch, train_acc_list=meters["train_acc"].acc)

    model.load_state_dict(torch.load(best_model_path, map_location=device)['state_dict'])
    test_matrix_list, scores, true_labels = test(args, vali_loader, model, device)
    snr_accuracies, total_accuracy, classwise_accuracies = compute_accuracies(test_matrix_list)
    args.logger.info(f"Test Acc: {total_accuracy:.4f}")
    result_generator.plot_acc_of_dif_snr(snr_accuracies)
    result_generator.plot_classwise_acc_of_dif_snr(classwise_accuracies)
    result_generator.plot_confusion_matrix(test_matrix_list)
    result_generator.visualize_tsne(scores, true_labels)
    return total_accuracy


def validation(args, loader, model, criterion, device):
    model.eval()
    loss_list = []
    vali_acc = {"current": 0, "total": 0}
    with torch.no_grad():
        for i, (batch_x, batch_y, _) in enumerate(loader):
            if type(batch_x) == list:
                batch_x = (x.to(device) for x in batch_x)
            else:
                batch_x = batch_x.to(device)
            true = batch_y.to(device)
            predict = model(batch_x)
            loss = criterion(predict, true)
            predict = torch.argmax(predict, 1)
            vali_acc["current"] += (predict == true).sum().item()
            vali_acc["total"] += len(batch_y)
            loss_list.append(loss.detach().item())
    return np.mean(loss_list), vali_acc["current"] / vali_acc["total"]


def test(args, loader, model, device):
    model.eval()
    scores, true_labels = [], []
    matrix_list = [np.zeros(shape=(len(args.choose_classes), len(args.choose_classes)), dtype=np.int32) for _ in args.choose_snrs]
    snr_to_idx = {snr: idx for idx, snr in enumerate(args.choose_snrs)}
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_snr) in enumerate(loader):
            if type(batch_x) == list:
                batch_x = (x.to(device) for x in batch_x)
            else:
                batch_x = batch_x.to(device)
            true = batch_y.numpy()
            true_labels.extend(true)
            predict = model(batch_x)
            scores.extend(predict.cpu().numpy())
            predict = torch.argmax(predict, 1)
            for t, p, s in zip(true, predict.cpu().numpy(), batch_snr.numpy()):
                snr_idx = snr_to_idx.get(s)
                matrix_list[snr_idx][t, p] += 1
    return matrix_list, scores, true_labels


def compute_accuracies(matrix_list):
    num_classes = matrix_list[0].shape[0]
    num_snr = len(matrix_list)
    snr_accuracies = []
    classwise_accuracies = np.zeros((num_classes, num_snr))

    for snr_idx, matrix in enumerate(matrix_list):
        correct = np.trace(matrix)
        total = matrix.sum()
        acc = correct / total if total > 0 else 0.0
        snr_accuracies.append(acc)
        for cls in range(num_classes):
            cls_total = matrix[cls, :].sum()
            cls_correct = matrix[cls, cls]
            cls_acc = cls_correct / cls_total if cls_total > 0 else 0.0
            classwise_accuracies[cls, snr_idx] = cls_acc

    total_matrix = np.sum(matrix_list, axis=0)
    total_correct = np.trace(total_matrix)
    total_samples = total_matrix.sum()
    total_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return snr_accuracies, total_accuracy, classwise_accuracies


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # pytorch cpu seed
    torch.manual_seed(seed)
    # pytorch gpu seed
    torch.cuda.manual_seed(seed)
    # pytorch multiple gpu seed
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automatic Modulation Classification')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
    parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
    parser.add_argument('--seed', type=int, default=random.randint(0, 2 ** 32 - 1), help='experiment seed')
    parser.add_argument('--itr', type=int, default=1, help='times of experiment')
    parser.add_argument('--result_root_path', type=str, default='./results', help='location to store train results')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--print_freq', default=50, type=int, metavar='N', help='print frequency (default: 10)')

    # Dataset parameters
    parser.add_argument('--dataset_name', type=str, default='RML2016.10a', help='chose dataset')
    parser.add_argument('--data_split', type=str, default='0.6,0.2,0.2',
                        help='train/val/test split, must be ratio')
    parser.add_argument('--data_augmentation', type=str, default='', help='data augmentation method(RSC/SSC)')
    parser.add_argument('--data_augmentation_params', type=str, default='', help='data augmentation method paramiters')

    # Model parameters
    parser.add_argument('--model', type=str, default='PETCGDNN', help='chose model')
    parser.add_argument('--resume', type=str, default="", help='path to latest checkpoint (default: none)')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--weight_decay', type=float, default=0.001, metavar='LR', help='weight decay (default: 0.05)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--lradj', type=str, default='type1', metavar='LR', help='adjust learning rate')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--use_scaler', type=bool, default=True, help='Enable GradScaler for mixed precision training')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=1, help='gpu')

    args = parser.parse_args()
    args.lr = args.blr * args.batch_size / 256
    args.start_epoch = 0
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.data_split = parse_string_to_list(args.data_split, "float")
    assert sum(args.data_split) == 1 and args.data_split[0] > 0 and args.data_split[1] > 0 and len(args.data_split) == 3
    if args.data_augmentation != '':
        args.data_augmentation_params = parse_string_to_list(args.data_augmentation_params, "float")

    args.choose_classes = None
    args.choose_snrs = None

    init_experiment(args)
    args.logger.info(vars(args))

    setup_seed(args.seed)
    args.logger.info(f"Init seed:{args.seed}")

    top1_acc = {'avg': .0, 'max': .0, 'min': .0, 'all': [], 'itr': args.itr}
    experiment_acc = []
    for i_experiment_time in range(args.itr):
        args.logger.info(f'>>>>>>>start training : times_{i_experiment_time}>>>>>>>>>>>>>>>>>>>>>>>>>>')
        train(args, i_experiment_time)
