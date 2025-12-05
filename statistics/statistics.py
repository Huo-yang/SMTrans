import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_matrix_result(path, size_limit = 5):
    list_dir = os.listdir(path)
    list_dir = [d for d in list_dir if os.path.isdir(os.path.join(path, d))]
    assert len(list_dir) >= size_limit, f"Results has fewer than {size_limit}."
    print(f"Total {len(list_dir)} results, now cut {size_limit}.")
    list_dir = list_dir[:size_limit]
    list_dir = [os.path.join(path, f, "0_matrix", "0_confusion_matrixs.npy") for f in list_dir]
    matrix_dict = {f: np.load(f) for f in list_dir}
    return matrix_dict


def get_top_matrices(matrix_dict, top):
    assert len(matrix_dict) >= top, f"Dictionary has fewer than {top}."
    accuracies = []
    for key, matrix in matrix_dict.items():
        total_conf_matrix = np.sum(matrix, axis=0)
        total_samples = np.sum(total_conf_matrix)
        correct_predictions = np.trace(total_conf_matrix)
        accuracy = correct_predictions / total_samples
        accuracies.append((key, accuracy))
    sorted_accuracies = sorted(accuracies, key=lambda x: x[1], reverse=True)[:top]
    print(f"Top {top} accuracy dict:{sorted_accuracies}.")
    top_matrices_dict = dict((key, matrix_dict[key]) for key, _ in sorted_accuracies)
    # print(top_matrices_dict)
    return top_matrices_dict


def get_snr_table(matrix_dict, path):
    accuracy = {"name": ["correct", "total", "ratio"] * len(matrix_dict)}
    db_list = list(range(-20, 20, 2))
    for db in db_list:
        accuracy[f"{db}db"] = []
    for idx, matrix in enumerate(matrix_dict.values()):
        for db_idx, snr_matrix in enumerate(matrix):
            matrix_trace = np.trace(snr_matrix)
            matrix_sum = np.sum(snr_matrix)
            accuracy[f"{db_list[db_idx]}db"].append(matrix_trace)
            accuracy[f"{db_list[db_idx]}db"].append(matrix_sum)
            accuracy[f"{db_list[db_idx]}db"].append(matrix_trace / matrix_sum)
    df = pd.DataFrame(accuracy)
    df.to_excel(path, index=False)
    print(f"Save to {path}.")


def get_class_table(matrix, path):
    snr_values = np.arange(-20, 20, 2)
    n_classes = matrix.shape[1]
    accuracy_matrix = np.zeros((n_classes, len(snr_values)), dtype=float)
    for snr_idx in range(len(snr_values)):
        for cls in range(n_classes):
            total = matrix[snr_idx, cls, :].sum()
            if total > 0:
                accuracy_matrix[cls, snr_idx] = matrix[snr_idx, cls, cls] * 100 / total
            else:
                accuracy_matrix[cls, snr_idx] = np.nan
    df = pd.DataFrame(
        accuracy_matrix,
        index=[f"Class {i}" for i in range(n_classes)],
        columns=[f"SNR {s}" for s in snr_values]
    )
    df.to_excel(path, index=True)


def plot_confusion_matrix(confusion_matrix, classes=[], title='', cmap=plt.cm.Greens, save_filename=None):
    if classes == []:
        classes = [str(i) for i in range(len(confusion_matrix))]
    norm_confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True) * 100

    if len(classes) < 15:
        plt.figure(figsize=(10, 8))
    else:
        plt.figure(figsize=(18, 16))

    plt.imshow(norm_confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.title("accuracy: {:.2f}".format(np.trace(confusion_matrix)/np.sum(confusion_matrix)), fontsize=15)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=-45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    iters = np.reshape([[[i, j] for j in range(len(classes))] for i in range(len(classes))],
                        (confusion_matrix.size, 2))
    for i, j in iters:
        if norm_confusion_matrix[i, j] > 50:
            color = "white"
        else:
            color = "black"
        plt.text(j, i, "{:.1f}%".format(norm_confusion_matrix[i, j]), fontsize=12, va='bottom', ha='center',
                    color=color)
        plt.text(j, i, "{}".format(confusion_matrix[i, j]), fontsize=10, va='top', ha='center', color=color)

    plt.xlabel('Prediction', fontsize=13)
    plt.ylabel('Real label', fontsize=13)
    plt.tight_layout()
    # plt.show()
    if save_filename is not None:
        plt.savefig(save_filename, dpi=600, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    matrix_dict = load_matrix_result("../results/10a")
    matrix_dict = get_top_matrices(matrix_dict, 3)
    get_snr_table(matrix_dict, "output.xlsx")