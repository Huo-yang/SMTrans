import os
from statistics.statistics import *

class_name = ['8PSK', 'AM-DSB', 'BPSK', 'CPFSK', 'GFSK',
                    'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
size_limit = 20
top = 5
# 对比
dir_list = ["results/CE/10a/amcnet", "results/CE/10a/fea_t_128" , "results/CE/10a/iq_former_128", "results/CE/10a/MCLDNN", "results/CE/10a/PETCGDNN", "results/CE/10a/smt_128", 
            "results/CE/10b/amcnet", "results/CE/10b/fea_t_128", "results/CE/10b/iq_former_128", "results/CE/10b/MCLDNN", "results/CE/10b/PETCGDNN", "results/CE/10b/smt_128"]
# 消融
# dir_list = ["results/AE/RML2016.10a/smt_128_SPE", "results/AE/RML2016.10a/smt_128_OE" , "results/AE/RML2016.10a/smt_128_SAM", "results/AE/RML2016.10a/smt_128_MSA"]
# 数据增强
# dir_list = ["results/AGE/RML2016.10a/smt_128/random_phase_offset/-0.25pi,0.25pi", "results/AGE/RML2016.10a/smt_128/random_phase_offset/-0.5pi,0.5pi" , 
#             "results/AGE/RML2016.10a/smt_128/random_phase_offset/-0.75pi,0.75pi", "results/AGE/RML2016.10a/smt_128/random_phase_offset/-1pi,1pi", 
#             "results/AGE/RML2016.10a/smt_128/random_stretching/1,1.5"]
dir_list = ["results/CE/10a/smt_128"]
for statistics_path in dir_list:
    matrix_dict = load_matrix_result(statistics_path, size_limit)
    matrix_dict = get_top_matrices(matrix_dict, top)

    # get_snr_table(matrix_dict, os.path.join(statistics_path, "top_acc_table.xlsx"))

    # merge_matrix = np.zeros((20, len(class_name), len(class_name)), dtype=int)
    # for _, v in matrix_dict.items():
    #     merge_matrix += v
    
    # get_class_table(merge_matrix, os.path.join(statistics_path, "top_class_table.xlsx"))
    # continue
    # for i in range(merge_matrix.shape[0]):
    #     plot_confusion_matrix(merge_matrix[i], classes=class_name, save_filename=os.path.join(statistics_path, "top_matrix_{}db.png".format(i * 2 - 20)))

    # merge_all_db = np.zeros((len(class_name), len(class_name)), dtype=int)
    # for i in range(0, 20):
    #     merge_all_db += merge_matrix[i]
    # plot_confusion_matrix(merge_all_db, classes=class_name, save_filename=os.path.join(statistics_path, "top_matrix_all_db.png"))

    # row_sums = merge_all_db.sum(axis=1)
    # accs = np.divide(np.diag(merge_all_db), row_sums, out=np.zeros_like(row_sums, dtype=float), where=row_sums!=0)
    # accs = np.round(accs * 100, 2)
    # print(accs)

    