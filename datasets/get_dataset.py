from datasets.sig_data import get_dataset_precessed
from datasets.dataset_info import *

get_dataset_info = {
    'RML2016.04c': RML2016_04c,
    'RML2016.10a': RML2016_10a,
    'RML2016.10b': RML2016_10b,
    'RML2018.01a': RML2018_01a,
    'HisarMod2019.1': HisarMod2019_1,
}

def get_datasets(args):
    if args.dataset_name not in get_dataset_info.keys():
        raise ValueError
    data_info = get_dataset_info[args.dataset_name]
    train_dataset, vali_dataset, test_dataset = get_dataset_precessed(data_info, args)
    return train_dataset, vali_dataset, test_dataset