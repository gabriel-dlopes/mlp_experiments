#utils for training callbacks and schedules
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay

#get callbacks from config
def get_callbacks(cfg_train):
    callbacks = []
    if hasattr(cfg_train, "callbacks") and \
       hasattr(cfg_train.callbacks, "early_stopping") and \
       getattr(cfg_train.callbacks.early_stopping, "enable", False):
        es = cfg_train.callbacks.early_stopping
        callbacks.append(
            EarlyStopping(
                monitor=getattr(es, "monitor", "val_loss"),
                patience=getattr(es, "patience", 20),
                min_delta=getattr(es, "delta", 0.0),
                restore_best_weights=getattr(es, "restore_best_weights", True),
                verbose=1,
            )
        )
    return callbacks

#learning rate scheduler from config
def lr_scheduler(cfg_train):
    if getattr(cfg_train, "lr_strategy", "constant") == "exponential":
        return ExponentialDecay(
            initial_learning_rate=getattr(cfg_train, "initial_lr", 1e-2),
            decay_steps=getattr(cfg_train, "decay_step", 50),
            decay_rate=getattr(cfg_train, "decay_rate", 0.9),
            staircase=getattr(cfg_train, "stair", True),
        )
    if getattr(cfg_train, "lr_strategy", "constant") == "constant":
        return getattr(cfg_train, "lr", 1e-3)
    raise ValueError(f"unknown lr strategy: {getattr(cfg_train, 'lr_strategy', None)}")

#track prediction confidence per epoch
class ConfidencePerEpoch(Callback):
    def __init__(self, x_val, y_val_cat):
        super().__init__()
        self.x_val = x_val
        self.y_val_cat = y_val_cat
        self.conf_acertos = []
        self.conf_erros = []
        self.conf_erros_true = []

    def on_epoch_end(self, epoch, logs=None):
        y_true = np.argmax(self.y_val_cat, axis=1)
        y_pred_proba = self.model.predict(self.x_val, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        conf_hits, conf_err_true = [], []
        for prob, yp, yt in zip(y_pred_proba, y_pred, y_true):
            if yp == yt:
                conf_hits.append(-np.log(max(prob[yp], 1e-12)))
            else:
                conf_err_true.append(-np.log(max(prob[yt], 1e-12)))

        self.conf_acertos.append(np.mean(conf_hits) if conf_hits else 0.0)
        self.conf_erros_true.append(np.mean(conf_err_true) if conf_err_true else 0.0)

#track abs mean of weights per epoch
class WeightsAbsMeanPerEpoch(Callback):
    def __init__(self):
        super().__init__()
        self.layer_histories = {}

    def on_epoch_end(self, epoch, logs=None):
        for layer in self.model.layers: #model.layers is an object inside the class Callback
            if hasattr(layer, "get_weights"): #checks if the layer has trainable weights
                weights = layer.get_weights()
                if weights: #checks if the list is not empty
                    W = weights[0]
                    abs_mean = float(np.mean(np.abs(W)))
                    self.layer_histories.setdefault(layer.name, []).append(abs_mean)

#save full weight matrices at selected epochs
class WeightsMatrixPrinter(Callback):
    def __init__(self, output_dir, epochs_to_print=(50, 500, 1000)): #need to put this choice on callback config
        super().__init__()
        self.output_dir = output_dir
        self.epochs_to_print = set(int(e) for e in epochs_to_print)
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        ep = epoch + 1
        if ep not in self.epochs_to_print:
            return
        for layer in self.model.layers:
            if hasattr(layer, "get_weights"):
                weights = layer.get_weights()
                if weights:
                    W = weights[0]
                    fname = f"{layer.name}_weights_epoch{ep}.npy"
                    np.save(os.path.join(self.output_dir, fname), W)
