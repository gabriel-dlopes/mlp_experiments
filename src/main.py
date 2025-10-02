import os
import hydra
from omegaconf import DictConfig
import tensorflow as tf
import hydra.utils as hy_utils
from models import build_model
from data_loader import load_data
from utils import (
    get_callbacks,
    lr_scheduler,
    ConfidencePerEpoch,
    WeightsAbsMeanPerEpoch,
    WeightsMatrixPrinter
)
from experiment_logger import JSONLogger
from hydra.core.hydra_config import HydraConfig   
from run_plots import save_selected_plots
from omegaconf import OmegaConf


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    #load data and metadata
    (x_train, y_train_cat), (x_test, y_test_cat), meta = load_data(cfg.data)

    #build model (expects meta, not cfg.data)
    model = build_model(cfg.model, meta)

    #define optimizer and learning rate schedule
    lr = lr_scheduler(cfg.train)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    #compile model
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = ["accuracy"]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    #define hydra run directory and helpers
    #hydra sets cwd to the run folder (unique per execution)

    #use hydra's runtime output dir
    run_dir = HydraConfig.get().runtime.output_dir
    project_root = hy_utils.get_original_cwd()

    dataset = cfg.data.name
    model_tag = getattr(cfg.model, "name", "model")  # keep it simple

    #per-model index (all runs of this model on this dataset)
    per_model_index = os.path.join(project_root, "outputs", dataset, model_tag, "runs_index.jsonl")

    #per-dataset leaderboard (top runs across models)
    dataset_leaderboard = os.path.join(project_root, "outputs", dataset, "leaderboard.jsonl")

    #extract epochs for weight snapshots from config if present
    weights_snap_interval = getattr(cfg.train, "weight_snap_interval", 50)
    
    weights_snap_dir = os.path.join(run_dir, "weights_snapshots")
    os.makedirs(weights_snap_dir, exist_ok=True)

    #set base callbacks
    callbacks = get_callbacks(cfg.train)
    #confidence_cb = ConfidencePerEpoch(x_test, y_test_cat)
    #weights_mean_cb = WeightsAbsMeanPerEpoch()
    #weights_matrix_cb = WeightsMatrixPrinter(
    #    epochs_to_print=weights_snap_epochs,
    #    output_dir=weights_snap_dir
    #)

    #define absolute path for the global runs index (outside the run dir)
    #this makes the index accumulate across runs instead of being inside each run
    project_root = hy_utils.get_original_cwd()
    global_index = os.path.join(project_root, "outputs", "runs_index.jsonl")

    #attach json logger
    json_logger = JSONLogger(
        run_dir=run_dir,
        cfg=cfg,
        data_meta=meta,
        x_val=x_test,
        y_val_cat=y_test_cat,
        weights_snap_interval=weights_snap_interval,
        index_file=global_index
    )

    #register callbacks
    callbacks += [json_logger]

    #train model
    model.fit(
        x_train, y_train_cat,
        validation_data=(x_test, y_test_cat),
        epochs=cfg.train.epochs,
        batch_size=cfg.data.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    #post-train: optionally create per-run plots
    #hydra sets cwd to a unique run folder; ensure run_dir points to it (e.g., Path.cwd())
    print(getattr(cfg.plots, "enable", False))
    print("\n=== effective cfg ===")
    print(OmegaConf.to_yaml(cfg))
    print("type(cfg.plots):", type(getattr(cfg, "plots", None)))
    print("cfg.plots value:", getattr(cfg, "plots", None))
    if getattr(cfg, "plots", None) and getattr(cfg.plots, "enable", False):
        #build boolean toggles with safe defaults
        toggles = {
            "acc": bool(getattr(cfg.plots, "acc", True)),
            "loss": bool(getattr(cfg.plots, "loss", True)),
            "confidence": bool(getattr(cfg.plots, "confidence", True)),
        }   

        saved = save_selected_plots(run_dir, toggles)
        if saved:
            print("[plots] saved:", {k: os.path.relpath(v, run_dir) for k, v in saved.items()})
        else:
            print("[plots] nothing was generated (check data availability and toggles).")

if __name__ == "__main__":
    main()
