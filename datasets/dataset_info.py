RML2016_04c = {
    "name": "RML2016.04c",
    "path": "/media/ubuntu/SDR_HY/datasets/RML2016.04c/processed",
    "class_names": ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK',
                    'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK',
                    'WBFM'],
    "SNRs":list(range(-20, 20, 2)),
    "data_length": 128
        }

RML2016_10a = {
    "name": "RML2016.10a",
    "path": "/media/ubuntu/SDR_HY/datasets/RML2016.10a/processed",
    "processed": {
        "0.600.200.20": "/media/ubuntu/SDR_HY/datasets/RML2016.10a/T0.6V0.2T0.2Seed42",
    },
    "class_names": ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK',
                    'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK',
                    'WBFM'],
    "SNRs": list(range(-20, 20, 2)),
    "data_length": 128
}

RML2016_10b = {
    "name": "RML2016.10b",
    "path": "/media/ubuntu/SDR_HY/datasets/RML2016.10b/processed",
    "processed": {
        "0.600.200.20": "/media/ubuntu/SDR_HY/datasets/RML2016.10b/T0.6V0.2T0.2Seed42",
    },
    "class_names": ['8PSK', 'AM-DSB', 'BPSK', 'CPFSK', 'GFSK',
                    'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM'],
    "SNRs": list(range(-20, 20, 2)),
    "data_length": 128
}

RML2018_01a = {
    "name": "RML2018.01a",
    "path": "/media/ubuntu/SDR_HY/datasets/RML2018.01a/processed",
    "processed": {
        "0.600.200.20": "/media/ubuntu/SDR_HY/datasets/RML2018.01a/T0.6V0.2T0.2Seed42",
        "0.800.100.10": "/media/ubuntu/SDR_HY/datasets/RML2018.01a/T0.8V0.1T0.1Seed42",
    },
    "class_names": ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK',
                    '8PSK', '16PSK', '32PSK', '16APSK', '32APSK',
                    '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
                    '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC',
                    'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK'],
    "SNRs": list(range(-20, 32, 2)),
    "data_length": 1024
}

HisarMod2019_1 = {
    "name": "HisarMod2019.1",
    "path": "/media/ubuntu/SDR_HY/datasets/HisarMod2019.1/processed",
    "class_names": ['AM-DSB', 'AM-SC', 'AM-USB', 'AM-LSB', 'FM',
                    'PM', '2FSK', '4FSK', '8FSK', '16FSK',
                    '4PAM', '8PAM', '16PAM', 'BPSK', 'QPSK',
                    '8PSK', '16PSK', '32PSK', '64PSK', '4QAM',
                    '8QAM', '16QAM', '32QAM', '64QAM', '128QAM',
                    '256QAM'],
    "SNRs": list(range(-20, 20, 2)),
    "data_length": 1024
}