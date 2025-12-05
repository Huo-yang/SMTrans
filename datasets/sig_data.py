import numpy as np
import os
from copy import deepcopy
from torch.utils.data import Dataset
from scipy.signal import stft
from datasets.data_augmentation import FeatureAugmentor


def get_rml_data(root_dir:str, name: str, snr: int) -> np.ndarray:
    path_base = os.path.join(root_dir, name)
    file_name = "{}_{}db.npy".format(name, snr)
    data = np.load(os.path.join(path_base, file_name))
    return data


def aug_identity(data:np.ndarray)->np.ndarray:
    return data


class SigDatasetWithSNR(Dataset):
    def __init__(self, data_info:dict, args, role:str=None):
        self.role = role
        self.choose_classes = args.choose_classes
        self.choose_snrs = args.choose_snrs
        self.model = args.model
        self._load_data(data_info)
        args.logger.info(f"Loaded {len(self.sample_ids)} samples.")
        self.logger = args.logger
        self.aug = aug_identity
        self.aug_options = {
            "name": args.data_augmentation, 
            "params": args.data_augmentation_params
        }
        
    def init_augmentor(self):
        if self.aug_options["name"] is not "":
            config = {}
            config[self.aug_options["name"]] = {"start": self.aug_options["params"][0], "end": self.aug_options["params"][1]}
            prbe = 0.5
            self.aug = FeatureAugmentor(config, prbe)
            self.logger.info(f"Use data augmentation with prob={prbe}: {config}.")

    def _load_data(self, data_info):
        self.sigs_ = []
        self.names_ = []
        self.snrs_ = []
        for name in self.choose_classes:
            for snr in self.choose_snrs:
                single_name_snr = get_rml_data(data_info["path"], name, snr)
                self.sigs_.extend(single_name_snr)
                self.names_.extend([self.choose_classes.index(name)] * len(single_name_snr))
                self.snrs_.extend([snr] * len(single_name_snr))
        self.sigs_ = np.array(self.sigs_, dtype=np.float32)
        self.names_ = np.array(self.names_, dtype=np.int64)
        self.snrs_ = np.array(self.snrs_, dtype=np.int64)
        self.sample_ids = np.array(range(len(self.sigs_)), dtype=np.int64)

    def __len__(self):
        return len(self.sigs_)

    def __getitem__(self, idx):
        sig = self.sigs_[idx]
        if self.names_[idx] is not self.choose_classes.index("WBFM"):
            sig = self.aug(sig)
        if 'iq_former' in self.model:
            _, _, stp = stft(
                x=sig[0, :],
                fs=1.0,
                window='blackman',
                nperseg=31,
                noverlap=30,
                nfft=128)
            stp = np.array(stp, dtype=np.float32)
            return (sig, np.expand_dims(stp[:32, :], 0)), self.names_[idx], self.snrs_[idx]
        else:
            name = self.names_[idx]
            snr = self.snrs_[idx]
            return sig, name, snr


class SigDatasetWithSNRLoad(SigDatasetWithSNR):
    def __init__(self, role:str, data_info:dict, args):
        args.logger.info(f"Load {role} dataset.")
        self.ratio_str = ''.join([f"{x:.2f}" for x in args.data_split])
        super().__init__(data_info, args, role)

    def _load_data(self, data_info):
        base_path = data_info["processed"][self.ratio_str]
        self.sigs_ = np.load(os.path.join(base_path, f"{self.role}_sigs.npy"))
        self.names_ = np.load(os.path.join(base_path, f"{self.role}_labels.npy"))
        self.snrs_ = np.load(os.path.join(base_path, f"{self.role}_snrs.npy"))
        self.sample_ids = np.load(os.path.join(base_path, f"{self.role}_idxs.npy"))
        self.sigs_ = np.array(self.sigs_, dtype=np.float32)
        self.names_ = np.array(self.names_, dtype=np.int64)
        self.snrs_ = np.array(self.snrs_, dtype=np.int64)
        self.sample_ids = np.array(self.sample_ids, dtype=np.int64)


def subsample_with_ratio(dataset: SigDatasetWithSNR, ratio):
    remain_idx = []
    for name in dataset.choose_classes:
        name_index = dataset.choose_classes.index(name)
        for snr in dataset.choose_snrs:
            mask = (dataset.names_ == name_index) & (dataset.snrs_ == snr)
            index = np.where(mask)[0]
            num_sample = int(len(index) * ratio)
            selected = np.random.choice(index, size=num_sample, replace=False)
            remain_idx.extend(selected.tolist())
    dataset = subsample_dataset(dataset, remain_idx)
    return dataset


def subsample_dataset(dataset: SigDatasetWithSNR, idxs):
    dataset.sigs_ = dataset.sigs_[idxs]
    dataset.names_ = dataset.names_[idxs]
    dataset.snrs_ = dataset.snrs_[idxs]
    dataset.sample_ids = dataset.sample_ids[idxs]
    return dataset


def get_dataset(data_info, args):
    if args.choose_classes is None:
        args.choose_classes = data_info["class_names"]
    if args.choose_snrs is None:
        args.choose_snrs = data_info["SNRs"]
    whole_dataset = SigDatasetWithSNR(data_info, args)
    train_dataset = subsample_with_ratio(deepcopy(whole_dataset), ratio=args.data_split[0])
    other_indices = set(whole_dataset.sample_ids) - set(train_dataset.sample_ids)
    other_dataset = subsample_dataset(deepcopy(whole_dataset), np.array(list(other_indices)))

    vali_dataset = subsample_with_ratio(deepcopy(other_dataset), ratio=args.data_split[1] / sum(args.data_split[1:]))
    test_indices = set(other_dataset.sample_ids) - set(vali_dataset.sample_ids)
    test_dataset = subsample_dataset(deepcopy(whole_dataset), np.array(list(test_indices)))

    train_dataset.role = "train"
    vali_dataset.role = "vali"
    test_dataset.role = "test"
    train_dataset.init_augmentor()
    return train_dataset, vali_dataset, test_dataset


def get_dataset_precessed(data_info, args):
    if args.choose_classes is None:
        args.choose_classes = data_info["class_names"]
    if args.choose_snrs is None:
        args.choose_snrs = data_info["SNRs"]
    args.sig_size = data_info["data_length"]
    train_dataset = SigDatasetWithSNRLoad("train", data_info, args)
    vali_dataset = SigDatasetWithSNRLoad("vali", data_info, args)
    test_dataset = SigDatasetWithSNRLoad("test", data_info, args)
    train_dataset.init_augmentor()
    return train_dataset, vali_dataset, test_dataset