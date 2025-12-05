import numpy as np
import json
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import math
import pandas as pd
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
from statistics.statistics import *


class ResultGenerator:
    def __init__(self, path: str, class_names: list[str], snrs: list[int], times_now: int = None) -> None:
        self.path = path
        self.class_names = class_names
        self.snrs = snrs
        self.times_now = times_now

    def plot_loss(self, train_loss_list, start_epoch, end_epoch, validation_loss_list=None,
                  test_loss_list=None) -> None:
        fig, ax = plt.subplots()
        skip = max(1, int(len(train_loss_list) / (end_epoch - start_epoch) / 10))
        train_loss = np.array(train_loss_list).flatten()
        x1 = np.array(range(1, len(train_loss) + 1), dtype=np.float32) / len(train_loss_list[0]) + start_epoch
        plt.plot(x1[::skip], train_loss[::skip], color='#00BFFF', label="train loss")
        if validation_loss_list is not None:
            validation_loss_list = np.array(validation_loss_list).flatten()
            x2 = np.array(range(1, len(validation_loss_list) + 1), dtype=np.float32)
            plt.plot(x2, validation_loss_list, color='#00FF7F', label="validation loss")
        if test_loss_list is not None:
            average_test_loss_list = np.average(np.array(test_loss_list).T, axis=1)
            x3 = np.array(range(1, len(average_test_loss_list) + 1), dtype=np.float32) / self.len_test_loader
            plt.plot(x3[::math.ceil(skip / 2)], average_test_loss_list[::math.ceil(skip / 2)], color='#EF8A43',
                     label="test loss")
        plt.legend()
        plt.title(f"Loss")
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(math.ceil(end_epoch / 10)))
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig(os.path.join(self.path, "{}_loss.png".format(self.times_now if self.times_now != None else 0)))
        plt.close()

    def plot_acc(self, validation_acc_list, start_epoch, end_epoch, train_acc_list=None, test_acc_list=None,
                 name="") -> None:
        fig, ax = plt.subplots()
        if train_acc_list is not None:
            x1 = np.array(range(1, len(train_acc_list) + 1), dtype=np.int16) + start_epoch
            plt.plot(x1, train_acc_list, color='#00BFFF', label="train acc")
        x2 = np.array(range(1, len(validation_acc_list) + 1), dtype=np.int16)
        plt.plot(x2, validation_acc_list, color='#00FF7F', label="validation acc")
        if test_acc_list is not None:
            x3 = np.array(range(1, len(test_acc_list) + 1), dtype=np.int16)
            plt.plot(x3, test_acc_list, color='#EF8A43', label="test acc")
        plt.legend()
        plt.ylim((0, 1))
        plt.title("Acc")
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(math.ceil((end_epoch + 1) / 10)))
        plt.xlabel("epoch")
        plt.ylabel(name + "acc")
        plt.savefig(os.path.join(self.path, "{}_acc.png".format(self.times_now if self.times_now != None else 0)))
        plt.close()

    def plot_acc_of_dif_snr(self, test_list, validation_dict=None) -> None:
        plt.figure(figsize=(12.8, 9.6), dpi=100)
        plt.plot(self.snrs, test_list, color='#EF8A43', marker='*', ms=10, linewidth=4,
                 label="test accuracy")
        for x, y in zip(self.snrs, test_list):
            plt.text(x, y, f'{y:0.2f}', fontsize=14, color='red', ha='center', va='bottom')
        if validation_dict:
            vali_list = []
            for snr in self.snrs:
                vali_list.append(validation_dict["{}db".format(snr)][2])
            plt.plot(range(-20, 20, 2), vali_list, color='#00BFFF', marker='s', ms=10, linewidth=4,
                     label="validation accuracy")
        # for x, y in zip(range(-20, 20, 2), [sublist[index] for sublist in validation_acc_list]):
        #     plt.text(x, y, f'{y:0.2f}', fontsize=14, color='red', ha='center', va='bottom')
        plt.legend(fontsize="x-large", loc="lower left")
        plt.title("SNR vs ACC")
        plt.xlabel("SNR(db)")
        plt.ylabel("accuracy")
        plt.grid(True, which='major', axis='both', color='gray', linestyle='dashed')
        plt.ylim((0, 1))
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(2))
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
        plt.savefig(
            os.path.join(self.path, "{}_SNR_vs_ACC.png".format(self.times_now if self.times_now != None else 0)))
        plt.close()

    def plot_classwise_acc_of_dif_snr(self, classwise_acc):
        plt.figure(figsize=(12.8, 9.6), dpi=100)
        colors = plt.cm.get_cmap('tab20b', len(self.class_names))
        for cls_idx in range(len(self.class_names)):
            acc_list = classwise_acc[cls_idx]
            plt.plot(self.snrs, acc_list, marker='o', ms=8, linewidth=3,
                     label=f"{self.class_names[cls_idx]}", color=colors(cls_idx))
            # for x, y in zip(self.snrs, acc_list):
            #     plt.text(x, y, f'{y:0.2f}', fontsize=12, color=colors(cls_idx), ha='center', va='bottom')
        plt.legend(fontsize="large", loc="lower left")
        plt.title("SNR vs Class-wise ACC")
        plt.xlabel("SNR (dB)")
        plt.ylabel("Accuracy")
        plt.grid(True, which='major', axis='both', color='gray', linestyle='dashed')
        plt.ylim((0, 1))
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(2))
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
        filename = "{}_SNR_vs_Class_ACC.png".format(self.times_now if self.times_now is not None else 0)
        plt.savefig(os.path.join(self.path, filename))
        plt.close()

    def save_acc_of_dif_snr(self, dict):
        df = pd.DataFrame(dict)
        df.to_excel(
            os.path.join(self.path, "{}_SNR_vs_ACC.xlsx".format(self.times_now if self.times_now != None else 0)),
            index=False)

    def plot_confusion_matrix(self, confusion_matrixs, title='') -> None:
        path_dir = os.path.join(self.path, "{}_matrix".format(self.times_now if self.times_now != None else 0))
        os.mkdir(path_dir)
        np.save(
            os.path.join(path_dir, "{}_confusion_matrixs.npy".format(self.times_now if self.times_now != None else 0)),
            confusion_matrixs)
        for index, db in enumerate(self.snrs):
            plot_confusion_matrix(confusion_matrixs[index], classes=self.class_names,
                                         title=title, save_filename=os.path.join(path_dir, "{}db.png".format(db)))

    def visualize_tsne(self, features, true_labels):
        """
        使用t-SNE可视化样本分布，并用图例显示类别颜色，同时高亮低质量类样本
        """
        features = np.array(features)
        true_labels = np.array(true_labels)

        # t-SNE 降维
        tsne = TSNE(n_components=2, random_state=42)
        reduced_features = tsne.fit_transform(features)

        # 准备颜色
        colors = plt.cm.get_cmap('tab20b', len(self.class_names))

        plt.figure(figsize=(10, 8))

        # 每个类别分别绘图（用于 legend）
        for idx, cls_name in enumerate(self.class_names):
            cls_mask = true_labels == idx
            plt.scatter(
                reduced_features[cls_mask, 0],
                reduced_features[cls_mask, 1],
                color=colors(idx),
                label=cls_name,
                s=30,
                alpha=0.7
            )

        # 图例显示在图外右侧
        plt.legend(
            bbox_to_anchor=(1.02, 1),  # 横坐标略微向右偏移
            loc='upper left',  # 上边缘对齐
            borderaxespad=0.,
            title="Classes",
            fontsize='small',
            markerscale=1.5
        )
        plt.title('t-SNE of Samples')
        plt.axis('off')  # 关闭坐标轴
        plt.grid(False)  # 关闭网格线
        plt.tight_layout()
        plt.savefig(os.path.join(self.path, f"{self.times_now or 0}_t-SNE.png"))
        plt.close()
