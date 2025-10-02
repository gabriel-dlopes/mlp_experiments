#Script to create or upload the datasets
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import make_blobs

np.random.seed(42)

def make_simple(cfg):

    X, y = make_blobs(
            n_samples=cfg.n_samples,
            centers=[[0, 2], [-2, 0], [2, 0]],
            cluster_std=[0.9, 0.8, 0.8], 
            random_state=42)
    
    return X,y

def make_super_simple(cfg):

    X, y = make_blobs(
            n_samples=cfg.n_samples,
            centers=[[0, 2], [-2, 0], [2, 0]],
            cluster_std=[0.4, 0.4, 0.4], 
            random_state=42)
    
    return X, y

def make_complex(cfg):
        

    n_sample_clusters = cfg.n_sample_clusters
    n_sample_rings = cfg.n_sample_rings
    cluster_noise = cfg.cluster_noise

    #class 0 - two diagonal clusters
    c0_a = np.random.normal(loc=[-1, 1], scale=cluster_noise, size=(n_sample_clusters, 2))
    c0_b = np.random.normal(loc=[1, -1], scale=cluster_noise, size=(n_sample_clusters, 2))
    x0 = np.vstack((c0_a, c0_b))
    y0 = np.zeros(len(x0), dtype=int)

    #class 1 - two diagonal clusters
    c1_a = np.random.normal(loc=[1, 1], scale=cluster_noise, size=(n_sample_clusters, 2))
    c1_b = np.random.normal(loc=[-1, -1], scale=cluster_noise, size=(n_sample_clusters, 2))
    x1 = np.vstack((c1_a, c1_b))
    y1 = np.ones(len(x1), dtype=int)

    #class 2 - exterior ring
    r = np.random.uniform(2.2, 5.2, n_sample_rings)
    theta = np.random.uniform(0, 2*np.pi, n_sample_rings)
    x2 = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    y2 = np.full(len(x2), 2, dtype=int)

    X = np.vstack((x0, x1, x2))
    y = np.concatenate((y0, y1, y2))
    
    return X, y

def make_complex_5(cfg):

    n_sample_clusters = cfg.n_sample_clusters
    n_sample_rings = cfg.n_sample_rings
    cluster_noise = cfg.cluster_noise

    #class 0 — cluster 1 (up left)
    c0_a = np.random.normal(loc=[-1, 1], scale=cluster_noise, size=(n_sample_clusters, 2))
    y0 = np.zeros(len(c0_a), dtype=int)

    #class 1 — cluster 2 (low right)
    c0_b = np.random.normal(loc=[1, -1], scale=cluster_noise, size=(n_sample_clusters, 2))
    y1 = np.ones(len(c0_b), dtype=int)

    #class 2 — cluster 3 (up right)
    c1_a = np.random.normal(loc=[1, 1], scale=cluster_noise, size=(n_sample_clusters, 2))
    y2 = np.full(len(c1_a), 2, dtype=int)

    #class 3 — cluster 4 (low left)
    c1_b = np.random.normal(loc=[-1, -1], scale=cluster_noise, size=(n_sample_clusters, 2))
    y3 = np.full(len(c1_b), 3, dtype=int)

    #class 4 - exterior ring
    r = np.random.uniform(2.2, 5.2, n_sample_rings)
    theta = np.random.uniform(0, 2*np.pi, n_sample_rings)
    x4 = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    y4 = np.full(len(x4), 4, dtype=int)

    #join everything
    X = np.vstack((c0_a, c0_b, c1_a, c1_b, x4))
    y = np.concatenate((y0, y1, y2, y3, y4))

    return X, y

def make_complex_5_noise(cfg):

    n_sample_clusters = cfg.n_sample_clusters
    n_sample_rings = cfg.n_sample_rings
    cluster_noise = cfg.cluster_noise

    #class 0 — cluster 1 (up left)
    c0_a = np.random.normal(loc=[-1, 1], scale=cluster_noise, size=(n_sample_clusters, 2))
    y0 = np.zeros(len(c0_a), dtype=int)

    #class 1 — cluster 2 (low right)
    c0_b = np.random.normal(loc=[1, -1], scale=cluster_noise, size=(n_sample_clusters, 2))
    y1 = np.ones(len(c0_b), dtype=int)

    #class 2 — cluster 3 (up right)
    c1_a = np.random.normal(loc=[1, 1], scale=cluster_noise, size=(n_sample_clusters, 2))
    y2 = np.full(len(c1_a), 2, dtype=int)

    #class 3 — cluster 4 (low left)
    c1_b = np.random.normal(loc=[-1, -1], scale=cluster_noise, size=(n_sample_clusters, 2))
    y3 = np.full(len(c1_b), 3, dtype=int)

    #class 4 - exterior ring
    r = np.random.uniform(2.2, 5.2, n_sample_rings)
    theta = np.random.uniform(0, 2*np.pi, n_sample_rings)
    x4 = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    y4 = np.full(len(x4), 4, dtype=int)

    #join everything
    X = np.vstack((c0_a, c0_b, c1_a, c1_b, x4))
    y = np.concatenate((y0, y1, y2, y3, y4))

    return X, y

def make_mandala(cfg):

    # --- params (safe getters so struct mode won't crash) ---
    n_points = int(getattr(cfg, "n_points_circle", 500))
    contours = getattr(cfg, "contours", None)
    n_samples_per_cluster = getattr(cfg, "n_sample_clusters", None)
    if contours is None:
        # each tuple: (radius, n_waves, amplitude)
        contours = [
            (0.5, 5, 0.10),
            (2.3, 6, 0.12),
            (4.1, 7, 0.14),
            (5.9, 8, 0.16),
            (7.7, 9, 0.18),
        ]
    ring_radii = list(getattr(cfg, "ring_radii", [3.0, 5.0, 7.0]))
    ring_positions = int(getattr(cfg, "ring_positions", 12))
    cluster_std = float(getattr(cfg, "cluster_std", 0.10))
    n_samples_per_cluster = int(getattr(cfg, "n_samples_per_cluster", 30))
    n_classes = int(getattr(cfg, "num_classes", 5))
    seed = int(getattr(cfg, "seed", 0))

    rng = np.random.default_rng(seed)

    def wavy_circle(radius, n_waves, amp, n_points, phase=0.0):
        theta = np.linspace(0, 2 * np.pi, n_points)
        r = radius + amp * np.sin(n_waves * theta + phase)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.stack([x, y], axis=1)

    def generate_clusters(centers, n_classes, std=0.1, n_samples_per_cluster=30, base_seed=42):
        X_list, y_list = [], []
        for i, (cx, cy) in enumerate(centers):
            label = i % n_classes
            # single-center Gaussian blobs
            Xc, _ = make_blobs(
                n_samples=n_samples_per_cluster,
                centers=[(cx, cy)],
                cluster_std=std,
                random_state=base_seed + i
            )
            yc = np.full(n_samples_per_cluster, label, dtype=int)
            X_list.append(Xc)
            y_list.append(yc)
        return np.vstack(X_list), np.concatenate(y_list)

    # ----- main wavy contours -----
    X_main, y_main = [], []
    for i, (r, w, a) in enumerate(contours):
        pts = wavy_circle(radius=r, n_waves=w, amp=a, n_points=n_points, phase=i)
        labels = np.full(n_points, i % n_classes, dtype=int)
        X_main.append(pts)
        y_main.append(labels)
    X_main = np.vstack(X_main)
    y_main = np.concatenate(y_main)

    # ----- decorative clusters on rings -----
    angles = np.linspace(0, 2 * np.pi, ring_positions, endpoint=False)
    centers = [(rr * np.cos(ang), rr * np.sin(ang)) for rr in ring_radii for ang in angles]
    X_clusters, y_clusters = generate_clusters(
        centers, n_classes=n_classes, std=cluster_std,
        n_samples_per_cluster=n_samples_per_cluster, base_seed=42
    )

    # ----- join -----
    X = np.vstack([X_main, X_clusters])
    y = np.concatenate([y_main, y_clusters])

    return X, y

def make_mandala_noise(cfg):

    # --- params (safe getters so struct mode won't crash) ---
    n_points = int(getattr(cfg, "n_points_circle", 500))
    contours = getattr(cfg, "contours", None)
    n_samples_per_cluster = getattr(cfg, "n_sample_clusters", None)
    if contours is None:
        # each tuple: (radius, n_waves, amplitude)
        contours = [
            (0.5, 5, 0.10),
            (2.3, 6, 0.12),
            (4.1, 7, 0.14),
            (5.9, 8, 0.16),
            (7.7, 9, 0.18),
        ]
    ring_radii = list(getattr(cfg, "ring_radii", [3.0, 5.0, 7.0]))
    ring_positions = int(getattr(cfg, "ring_positions", 12))
    cluster_std = float(getattr(cfg, "cluster_std", 0.10))
    n_samples_per_cluster = int(getattr(cfg, "n_samples_per_cluster", 30))
    n_classes = int(getattr(cfg, "num_classes", 5))
    seed = int(getattr(cfg, "seed", 0))

    rng = np.random.default_rng(seed)

    def wavy_circle(radius, n_waves, amp, n_points, phase=0.0):
        theta = np.linspace(0, 2 * np.pi, n_points)
        r = radius + amp * np.sin(n_waves * theta + phase)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.stack([x, y], axis=1)

    def generate_clusters(centers, n_classes, std=0.3, n_samples_per_cluster=30, base_seed=42):
        X_list, y_list = [], []
        for i, (cx, cy) in enumerate(centers):
            label = i % n_classes
            # single-center Gaussian blobs
            Xc, _ = make_blobs(
                n_samples=n_samples_per_cluster,
                centers=[(cx, cy)],
                cluster_std=std,
                random_state=base_seed + i
            )
            yc = np.full(n_samples_per_cluster, label, dtype=int)
            X_list.append(Xc)
            y_list.append(yc)
        return np.vstack(X_list), np.concatenate(y_list)

    # ----- main wavy contours -----
    X_main, y_main = [], []
    for i, (r, w, a) in enumerate(contours):
        pts = wavy_circle(radius=r, n_waves=w, amp=a, n_points=n_points, phase=i)
        pts += np.random.normal(0, 0.1, size=pts.shape)  #adding noise to waky contours
        labels = np.full(n_points, i % n_classes, dtype=int)
        X_main.append(pts)
        y_main.append(labels)
    X_main = np.vstack(X_main)
    y_main = np.concatenate(y_main)

    # ----- decorative clusters on rings -----
    angles = np.linspace(0, 2 * np.pi, ring_positions, endpoint=False)
    centers = [(rr * np.cos(ang), rr * np.sin(ang)) for rr in ring_radii for ang in angles]
    X_clusters, y_clusters = generate_clusters(
        centers, n_classes=n_classes, std=cluster_std,
        n_samples_per_cluster=n_samples_per_cluster, base_seed=42
    )

    # ----- join -----
    X = np.vstack([X_main, X_clusters])
    y = np.concatenate([y_main, y_clusters])

    return X, y