#script to treat datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import numpy as np
from collections import Counter

from datasets import (make_simple, 
                      make_super_simple, 
                      make_complex,  
                      make_complex_5, 
                      make_complex_5_noise,
                      make_mandala,
                      make_mandala_noise
)
#registry with all datasets and its according source function

_DATASETS = {
    "simple": make_simple,
    "super_simple":make_super_simple,
    "complex": make_complex,
    "complex_5": make_complex_5,
    "complex_5_noise": make_complex_5_noise,
    "mandala": make_mandala,
    "mandala_noise": make_mandala_noise
}


def load_data(cfg_data):
    """
    single entry point called in main.py
    """

    #check dataset exists and load it
    name = cfg_data.name
    if name not in _DATASETS:
        raise ValueError(f"dataset '{name}' not found.") 
    X, y = _DATASETS[name](cfg_data)

    #split train and test
    stratify = y if getattr(cfg_data.split, "stratify", True) else None
    x_train, x_test, y_train, y_test = train_test_split(
        X, y,
        test_size=getattr(cfg_data.split, "test_size", 0.2),
        random_state=getattr(cfg_data.split, "random_state", 1337),
        stratify=stratify
    )

    #standardize features
    if getattr(cfg_data.preprocessing, "standardize", True):
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

    #convert labels to one-hot
    num_classes = int(np.max(y) + 1)
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    #build metadata
    meta = {
        "name": cfg_data.name,
        "input_dim": x_train.shape[1],
        "num_classes": num_classes,
        "data_name": name,
        "class_counts": {
            "train": dict(Counter(y_train)),
            "test": dict(Counter(y_test)),
        }
    }

    #return processed data and metadata
    return (x_train, y_train_cat), (x_test, y_test_cat), meta

