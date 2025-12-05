import argparse
import torch
import models
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pandas as pd
from collections import Counter
from utils.tools import *
from torch.utils.data import DataLoader, TensorDataset
from utils.general import init_experiment
from datasets.get_dataset import get_datasets

def find_weight_from_matrix_path(matrix_path):
    """
    输入:
        matrix_path

    返回:
        对应目录下的 pth.tar 权重文件路径
    """
    dir_root = matrix_path
    tar_files = glob.glob(os.path.join(dir_root, "*.tar")) + \
                glob.glob(os.path.join(dir_root, "*.pth.tar"))
    assert len(tar_files) > 0, f"No weight file found in {dir_root}"
    return tar_files[0]   # 通常每个目录只有一个


def two_model_battle(loder, base_model_path, battle_model_path, args):
    device = torch.device('cuda:{}'.format(args.gpu))
    
    model_base = getattr(models, "{}".format(args.model))(num_classes=len(args.choose_classes), sig_size=args.sig_size)
    model_battle = getattr(models, "{}".format(args.model))(num_classes=len(args.choose_classes), sig_size=args.sig_size)
    
    model_base.to(device)
    model_battle.to(device)

    ckpt_base = torch.load(base_model_path, weights_only=True, map_location=device)
    ckpt_battle = torch.load(battle_model_path, weights_only=True, map_location=device)
    model_base.load_state_dict(ckpt_base['state_dict'])
    model_battle.load_state_dict(ckpt_battle['state_dict'])

    model_base.eval()
    model_battle.eval()

    y_true, y_base, y_battle = [], [], []

    with torch.no_grad():
        for x, y, _ in loder:
            x, y = x.to(device), y.to(device)

            out_base = model_base(x)
            out_battle  = model_battle(x)

            y_true.append(y.cpu())
            y_base.append(out_base.argmax(1).cpu())
            y_battle.append(out_battle.argmax(1).cpu())

    y_true = torch.cat(y_true)
    y_base = torch.cat(y_base)
    y_battle = torch.cat(y_battle)

    return y_true, y_base, y_battle


def aggregate_multiple_models(test_loader, base_list, rsc_list, args):
    """
    输入两个等长 list：
        base_list[i] = base 模型的 confusion_matrix 路径
        rsc_list[i]  = rsc  模型的 confusion_matrix 路径

    自动找各自 tar 权重文件，并对所有模型循环推理
    最终累积误分统计总结果
    """

    focus_cls = args.focus_class
    focus_id = args.choose_classes.index(focus_cls)

    # 初始化累积向量
    base_p1_full_acc = np.zeros(len(args.choose_classes))
    rsc_p1_full_acc  = np.zeros(len(args.choose_classes))
    base_p2_full_acc = np.zeros(len(args.choose_classes))
    rsc_p2_full_acc  = np.zeros(len(args.choose_classes))

    for base_matrix, rsc_matrix in zip(base_list, rsc_list):

        # 自动找到权重文件
        base_weight = find_weight_from_matrix_path(base_matrix)
        rsc_weight  = find_weight_from_matrix_path(rsc_matrix)

        # 一个模型跑一次
        y_true, y_base, y_rsc = two_model_battle(test_loader,
                                                 base_weight,
                                                 rsc_weight,
                                                 args)

        # 误分 mask
        mask_focus = (y_true == focus_id)
        base_wrong_focus = y_base[mask_focus & (y_base != focus_id)]
        rsc_wrong_focus  = y_rsc[mask_focus & (y_rsc != focus_id)]

        base_wrong_others = y_true[(y_true != focus_id) & (y_base == focus_id)]
        rsc_wrong_others  = y_true[(y_true != focus_id) & (y_rsc == focus_id)]

        # 数量统计
        def count(vec):
            cnt = Counter(vec.tolist())
            return np.array([cnt.get(i, 0) for i in range(len(args.choose_classes))])

        base_p1_full_acc += count(base_wrong_focus)
        rsc_p1_full_acc  += count(rsc_wrong_focus)
        base_p2_full_acc += count(base_wrong_others)
        rsc_p2_full_acc  += count(rsc_wrong_others)

    return base_p1_full_acc, rsc_p1_full_acc, base_p2_full_acc, rsc_p2_full_acc



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automatic Modulation Classification')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
    parser.add_argument('--result_root_path', type=str, default='./results/test', help='location to store train results')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--dataset_name', type=str, default='RML2016.10a', help='chose dataset')
    parser.add_argument('--data_split', type=str, default='0.6,0.2,0.2',
                        help='train/val/test split, must be ratio')
    parser.add_argument('--data_augmentation', type=str, default='', help='data augmentation method(RSC/SSC)')
    parser.add_argument('--data_augmentation_params', type=str, default='', help='data augmentation method paramiters')
    parser.add_argument('--model', type=str, default='smt_128', help='chose model')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=1, help='gpu')

    parser.add_argument('--focus_class', type=str, default='WBFM',
                    help='the class to analyze misclassification behavior')
    
    args = parser.parse_args()
    args.data_split = parse_string_to_list(args.data_split, "float")
    assert sum(args.data_split) == 1 and args.data_split[0] > 0 and args.data_split[1] > 0 and len(args.data_split) == 3

    args.choose_classes = None
    args.choose_snrs = None

    init_experiment(args)
    args.logger.info(vars(args))

    _, _, test_dataset = get_datasets(args)
    test_dataset_size = len(test_dataset)
    args.logger.info(f"Test dataset size = {test_dataset_size}")

    focus_cls = args.focus_class
    assert focus_cls in args.choose_classes, f"{focus_cls} not in class list!"
    focus_id = args.choose_classes.index(focus_cls)

    test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    drop_last=False)

    base_list = [
    "results/CE/10a/smt_128/RML2016.10a_model_smt_128_date_20250701215750",
    "results/CE/10a/smt_128/RML2016.10a_model_smt_128_date_20250701213845",
    "results/CE/10a/smt_128/RML2016.10a_model_smt_128_date_20250701214755",
    "results/CE/10a/smt_128/RML2016.10a_model_smt_128_date_20250701220503",
    "results/CE/10a/smt_128/RML2016.10a_model_smt_128_date_20250701223221"
    ]

    rsc_list = [
    "results/AGE/RML2016.10a/smt_128/random_phase_offset/-1pi,1pi/RML2016.10a_model_smt_128_date_20251008160910",
    "results/AGE/RML2016.10a/smt_128/random_phase_offset/-1pi,1pi/RML2016.10a_model_smt_128_date_20251008162302",
    "results/AGE/RML2016.10a/smt_128/random_phase_offset/-1pi,1pi/RML2016.10a_model_smt_128_date_20251008143959",
    "results/AGE/RML2016.10a/smt_128/random_phase_offset/-1pi,1pi/RML2016.10a_model_smt_128_date_20251008154402",
    "results/AGE/RML2016.10a/smt_128/random_phase_offset/-1pi,1pi/RML2016.10a_model_smt_128_date_20251008140317"
    ]


    y_true, y_base, y_rsc = two_model_battle(test_loader, 
                     "results/CE/10a/smt_128/RML2016.10a_model_smt_128_date_20250701215750/0_smt_128_A[0.6341]_L[1.0203].pth.tar", 
                     "results/AGE/RML2016.10a/smt_128/random_phase_offset/-1pi,1pi/RML2016.10a_model_smt_128_date_20251008160910/0_smt_128_A[0.6388]_L[1.0052].pth.tar", 
                     args)

    (
    base_p1_full,
    rsc_p1_full,
    base_p2_full,
    rsc_p2_full
    ) = aggregate_multiple_models(test_loader, base_list, rsc_list, args)

    effective_test_size = len(base_list) * test_dataset_size
    base_p1_full = base_p1_full / effective_test_size
    rsc_p1_full  = rsc_p1_full  / effective_test_size
    base_p2_full = base_p2_full / effective_test_size
    rsc_p2_full  = rsc_p2_full  / effective_test_size

    base_p1_full *= 100
    rsc_p1_full  *= 100
    base_p2_full *= 100
    rsc_p2_full  *= 100

    # ---- 构建新的 x 轴顺序（排除 focus，再添加 "Total"） ----
    class_list = args.choose_classes.copy()
    focus_cls = args.focus_class
    focus_id  = args.choose_classes.index(focus_cls)

    other_ids   = [i for i in range(len(class_list)) if i != focus_id]
    other_names = [class_list[i] for i in other_ids]

    # ---- focus → others (probability, %)
    base_p1 = base_p1_full[other_ids]
    rsc_p1  = rsc_p1_full[other_ids]

    base_total1 = base_p1.sum()
    rsc_total1  = rsc_p1.sum()

    base_p1 = np.append(base_p1, base_total1)
    rsc_p1  = np.append(rsc_p1,  rsc_total1)
    x_labels_1 = other_names + ["Total"]

    # ---- others → focus (probability, %)
    base_p2 = base_p2_full[other_ids]
    rsc_p2  = rsc_p2_full[other_ids]

    base_total2 = base_p2.sum()
    rsc_total2  = rsc_p2.sum()

    base_p2 = np.append(base_p2, base_total2)
    rsc_p2  = np.append(rsc_p2,  rsc_total2)
    x_labels_2 = other_names + ["Total"]

    # ---------- 绘图 ----------
    width = 0.35
    x = np.arange(len(x_labels_1))

    # =============================== #
    # 图 1：True = focus → Pred ≠ focus
    # =============================== #
    plt.figure(figsize=(11, 4.5))

    plt.bar(x - width/2, base_p1, width, label='Baseline Model', color='steelblue')
    plt.bar(x + width/2, rsc_p1,  width, label='RSC Enhanced Model', color='tomato')

    plt.xlabel(f"Predicted Class (excluding '{focus_cls}') + Total", fontsize=11)
    plt.ylabel("Misclassification Rate (%)", fontsize=11)
    plt.title(f"Misclassification of '{focus_cls}' into Other Classes", fontsize=12)

    plt.xticks(x, x_labels_1, rotation=35, ha="right")
    plt.legend(fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig1_path = f'{args.result_root_path}/fig_focus_to_others_{focus_cls}.pdf'
    plt.savefig(fig1_path)


    # =============================== #
    # 图 2：True ≠ focus → Pred = focus
    # =============================== #
    plt.figure(figsize=(11, 4.5))

    plt.bar(x - width/2, base_p2, width, label='Baseline Model', color='steelblue')
    plt.bar(x + width/2, rsc_p2,  width, label='RSC Enhanced Model', color='tomato')

    plt.xlabel(f"True Class (excluding '{focus_cls}') + Total", fontsize=11)
    plt.ylabel("Misclassification Rate (%)", fontsize=11)
    plt.title(f"Misclassification of Other Classes into '{focus_cls}'", fontsize=12)

    plt.xticks(x, x_labels_2, rotation=35, ha="right")
    plt.legend(fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig2_path = f'{args.result_root_path}/fig_others_to_focus_{focus_cls}.pdf'
    plt.savefig(fig2_path)

    print(f"Figures saved:\n - {fig1_path}\n - {fig2_path}")


    # ---------- Excel 输出 ----------
    excel_path = os.path.join(args.result_root_path, f"misclassification_summary_{focus_cls}.xlsx")

    df1 = pd.DataFrame({
        "Class": x_labels_1,
        "Baseline (%)": base_p1,
        "RSC (%)": rsc_p1
    })

    df2 = pd.DataFrame({
        "Class": x_labels_2,
        "Baseline (%)": base_p2,
        "RSC (%)": rsc_p2
    })

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df1.to_excel(writer, sheet_name='focus_to_others', index=False)
        df2.to_excel(writer, sheet_name='others_to_focus', index=False)

    print(f"\nExcel file saved: {excel_path}")
