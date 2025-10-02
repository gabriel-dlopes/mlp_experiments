import os
import json
import time
from typing import Any, Dict, Optional
import numpy as np
from omegaconf import OmegaConf
from tensorflow.keras.callbacks import Callback

#convert objects (including dict KEYS) into JSON-safe Python types
def _sanitize_for_json(obj):
    #dict: fix keys + recurse into values
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            #fix keys first
            if isinstance(k, np.integer):
                k = int(k)
            elif isinstance(k, np.floating):
                k = float(k)
            elif isinstance(k, (bytes, bytearray)):
                k = k.decode("utf-8", errors="ignore")
            elif not isinstance(k, (str, int, float, bool, type(None))):
                k = str(k)  # last resort
            out[k] = _sanitize_for_json(v)
        return out

    #list/tuple: recurse
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]

    #numpy array -> list
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    #numpy scalars -> builtin
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)

    return obj

#json serialization for numpy and hydra objects
def _to_serializable(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

#atomic json write
def _write_json(path: str, payload: Dict[str, Any]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    safe = _sanitize_for_json(payload)  # <-- sanitize keys and values
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(safe, f, ensure_ascii=False, indent=2)  # default no longer needed
    os.replace(tmp, path)

def _get_model_architecture(units):
    # try to extract 'units' from config
    if units is not None:
        # handle both single int and list
        if isinstance(units, (list, tuple)):
            return [int(u) for u in units]
        else:
            return [int(units)]
        
class JSONLogger(Callback):
    def __init__(
        self,
        run_dir: str,
        cfg,
        data_meta: Dict[str, Any],
        x_val: Optional[np.ndarray] = None,
        y_val_cat: Optional[np.ndarray] = None,
        weights_snap_interval: Optional[int] = None,
        index_file: str = "outputs/runs_index.jsonl",
        leaderboard_file: Optional[str] = None
    ):
        super().__init__()
        self.run_dir = run_dir
        self.cfg = cfg
        self.model_arch = _get_model_architecture(getattr(cfg.model, "units", None))
        self.data_meta = data_meta
        self.x_val = x_val
        self.y_val_cat = y_val_cat
        self.weights_snap_interval = weights_snap_interval
        self.epoch_logs = []
        self.index_file = index_file
        self.leaderboard_file = leaderboard_file


        #paths
        self.paths = {
            "config": os.path.join(run_dir, "config.json"),
            "data": os.path.join(run_dir, "data_meta.json"),
            "model": os.path.join(run_dir, "model.json"),
            "history": os.path.join(run_dir, "history.json"),
            "epochs": os.path.join(run_dir, "epochs.jsonl"),
            "weights_dir": os.path.join(run_dir, "weights_snapshots"),
            "artifacts": os.path.join(run_dir, "artifacts"),
            "run_info": os.path.join(run_dir, "run_info.json"),
        }

    #on_train_begin: save config, dataset meta, model, run_info
    def on_train_begin(self, logs=None):
        cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
        _write_json(self.paths["config"], {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cfg": cfg_dict
        })

        _write_json(self.paths["data"], self.data_meta)

        model_conf = self.model.get_config()
        trainable = self.model.count_params()
        non_trainable = sum([w.count_params() for w in self.model.weights if not w.trainable])
        total_params = trainable + non_trainable
        _write_json(self.paths["model"], {
            "model_config": model_conf,
            "total_params": int(total_params),
            "trainable_params": int(trainable),
            "non_trainable_params": int(non_trainable)
        })

        os.makedirs(self.paths["weights_dir"], exist_ok=True)
        os.makedirs(self.paths["artifacts"], exist_ok=True)

        #model architecture

        #basic run info for quick search
        run_info = {
            "dataset": self.cfg.data.name,
            "model": self.model_arch,
            "run_id": os.path.basename(self.run_dir),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": self.data_meta.get("name"),
            "input_dim": self.data_meta.get("input_dim"),
            "num_classes": self.data_meta.get("num_classes"),
            "epochs": self.cfg.train.epochs,
            "batch_size": self.cfg.data.batch_size,
        }
        _write_json(self.paths["run_info"], run_info)

    #on_epoch_end: save per-epoch logs
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        ep1 = epoch + 1

        #confidence metrics (if validation data provided)
        conf_hits, conf_err_true = None, None
        if self.x_val is not None and self.y_val_cat is not None:
            y_true = np.argmax(self.y_val_cat, axis=1)
            proba = self.model.predict(self.x_val, verbose=0)
            y_pred = np.argmax(proba, axis=1)
            hits = [-np.log(max(proba[i, yp], 1e-12)) for i,(yp,yt) in enumerate(zip(y_pred,y_true)) if yp==yt]
            errt = [-np.log(max(proba[i, yt], 1e-12)) for i,(yp,yt) in enumerate(zip(y_pred,y_true)) if yp!=yt]
            conf_hits = float(np.mean(hits)) if hits else 0.0
            conf_err_true = float(np.mean(errt)) if errt else 0.0

        #layer weights absolute mean
        layer_abs_means = {}
        for layer in self.model.layers:
            if hasattr(layer, "get_weights"):
                w = layer.get_weights()
                if w:
                    W = w[0]
                    layer_abs_means[layer.name] = float(np.mean(np.abs(W)))

        #weights snapshot (every N epochs or at first/last)
        if isinstance(self.weights_snap_interval, int) and self.weights_snap_interval > 0:
            last_ep = int(self.cfg.train.epochs)
            if (ep1 % self.weights_snap_interval == 0) or (ep1 == 1) or (ep1 == last_ep):
                for layer in self.model.layers:
                    if hasattr(layer, "get_weights"):
                        w = layer.get_weights()
                        if w:
                            W = w[0]
                            npy_path = os.path.join(self.paths["weights_dir"], f"{layer.name}_epoch{ep1}.npy")
                            np.save(npy_path, W)
        elif isinstance(self.weights_snap_interval, (list, tuple, set)):
            if ep1 in self.weights_snap_interval:
                for layer in self.model.layers:
                    if hasattr(layer, "get_weights"):
                        w = layer.get_weights()
                        if w:
                            W = w[0]
                            npy_path = os.path.join(self.paths["weights_dir"], f"{layer.name}_epoch{ep1}.npy")
                            np.save(npy_path, W)


        record = {
            "epoch": ep1,
            "metrics": {k: float(v) for k,v in logs.items()},
            "confidence": {"nll_hits": conf_hits, "nll_err_true": conf_err_true},
            "weights_abs_mean": layer_abs_means,
            "timestamp": time.time()
        }
        self.epoch_logs.append(record)

        #append to jsonl
        with open(self.paths["epochs"], "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=_to_serializable) + "\n")

    #on_train_end: save final history and update runs_index
    def on_train_end(self, logs=None):
        #get final metrics and best val metric
        final = self.epoch_logs[-1]["metrics"] if self.epoch_logs else {}
        best_val = None
        if self.epoch_logs:
            vals = [e["metrics"].get("val_accuracy") for e in self.epoch_logs if "val_accuracy" in e["metrics"]]
            best_val = max(vals) if vals else None

        #model architecture
        model_arch = _get_model_architecture(getattr(self.cfg.model, "units", None))
        entry = {
            "run_id": os.path.basename(self.run_dir),
            "dataset": self.cfg.data.name,
            "model": self.model_arch,
            "batch_size": self.cfg.data.batch_size,
            "lr": getattr(self.cfg.train, "initial_lr", None),
            "final_val_acc": final.get("val_accuracy"),
            "best_val_acc": best_val,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        #append to per-model index
        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
        with open(self.index_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=_to_serializable) + "\n")

        #append to per-dataset leaderboard (optional)
        if self.leaderboard_file:
            os.makedirs(os.path.dirname(self.leaderboard_file), exist_ok=True)
            with open(self.leaderboard_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=_to_serializable) + "\n")
