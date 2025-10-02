import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import numpy as np

#load epochs.jsonl into tidy dataframe
def _load_epochs(run_dir):
    #build path
    path = os.path.join(run_dir, "epochs.jsonl")
    if not os.path.exists(path):
        return None
    df = pd.read_json(path, lines=True)
    m = pd.json_normalize(df["metrics"])
    m["epoch"] = df["epoch"].values
    #unpack confidence dict if present
    if "confidence" in df.columns:
        c = pd.json_normalize(df["confidence"])
        for col in c.columns:
            m[f"conf.{col}"] = c[col].values
    return m

#ensure artifacts dir
def _artifacts(run_dir):
    out = os.path.join(run_dir, "artifacts")
    os.makedirs(out, exist_ok=True)
    return out

#plot training vs validation accuracy for a single run
def save_acc_plot(run_dir):
    #load data
    m = _load_epochs(run_dir)
    if m is None: return None
    #plot
    plt.figure()
    if "accuracy" in m.columns:
        plt.plot(m["epoch"], m["accuracy"], label="train acc")
    if "val_accuracy" in m.columns:
        plt.plot(m["epoch"], m["val_accuracy"], label="val acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    #save
    out_dir = _artifacts(run_dir)
    out = os.path.join(out_dir, "acc.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out

#plot training vs validation loss for a single run
def save_loss_plot(run_dir):
    #load data
    m = _load_epochs(run_dir)
    if m is None: return None
    #plot
    plt.figure()
    if "loss" in m.columns:
        plt.plot(m["epoch"], m["loss"], label="train loss")
    if "val_loss" in m.columns:
        plt.plot(m["epoch"], m["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    #save
    out_dir = _artifacts(run_dir)
    out = os.path.join(out_dir, "loss.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out

#plot confidence summary (nll_hits vs nll_err_true)
def save_confidence_plot(run_dir):
    #load data
    m = _load_epochs(run_dir)
    if m is None: return None
    #columns are conf.nll_hits and conf.nll_err_true if logger wrote them
    hits = "conf.nll_hits" in m.columns
    errs = "conf.nll_err_true" in m.columns
    if not (hits or errs): return None
    #plot
    plt.figure()
    if hits: plt.plot(m["epoch"], m["conf.nll_hits"], label="nll hits (lower=better)")
    if errs: plt.plot(m["epoch"], m["conf.nll_err_true"], label="nll true on errors (lower=better)")
    plt.xlabel("epoch")
    plt.ylabel("nll")
    plt.legend()
    #save
    out_dir = _artifacts(run_dir)
    out = os.path.join(out_dir, "confidence.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out

#save per-run plots conditionally based on boolean toggles
def save_selected_plots(run_dir, toggles: dict):
    #expected keys: acc, loss, confidence, decision_boundary
    saved = {}

    #acc
    if toggles.get("acc", True):
        p = save_acc_plot(run_dir)
        if p: saved["acc"] = p

    #loss
    if toggles.get("loss", True):
        p = save_loss_plot(run_dir)
        if p: saved["loss"] = p

    #confidence
    if toggles.get("confidence", True):
        p = save_confidence_plot(run_dir)
        if p: saved["confidence"] = p

    return saved

