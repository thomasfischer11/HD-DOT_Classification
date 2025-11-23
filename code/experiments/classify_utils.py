# classify_utils_refactored.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
import xarray as xr
from typing import Dict, List, Optional, Tuple, Any
from scipy.stats import binom
import cedalion.imagereco.forward_model as fw
import os
import pandas as pd
import pickle
from matplotlib.patches import Patch

from configs import RunContext

def sp_map(sp):
    return f"{sp} cm"

int_map = {'01': '0.2 μM', '02': '0.4 μM', '03': '0.6 μM'}

SubjectStats = Dict[str, Dict[str, float]]  # {subject: {"mean": float, "std": float}}

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    

def preload_runs(ds, subjects, base_path, int_scaling, spatial_scaling):
    data, clean_map = {}, {}
    for si, subj in enumerate(subjects):
        print(f"Preloading data for subject {subj}...")
        data[subj], clean_map[subj] = {}, {}
        for run in range(ds.n_runs(si)):
            ep_rel = ds.epochs_labels_path(subj, run, int_scaling, spatial_scaling)
            data[subj][run] = load_pickle(os.path.join(base_path, ep_rel))
            clean_map[subj][run] = load_pickle(os.path.join(base_path, ds.clean_channels_path(subj, run)))
    return data, clean_map


def load_sma_sens_channels(base_path, data_type):
    name = {
        'BS_Laura': 'BS_Laura_SomMotA_sens_channels',
        'HD_Squeezing': 'HD_Squeezing_SomMotA_sens_channels'
    }.get(data_type)
    return load_pickle(os.path.join(base_path, name)) if name else None


def apply_sma_pruning(clean_map, sma_set):
    if not sma_set: return clean_map
    new_clean_map = {}
    for s in clean_map:
        new_clean_map[s] = {}
        for r in clean_map[s]:
            filt = [c for c in clean_map[s][r] if c in sma_set]
            new_clean_map[s][r] = filt
    return new_clean_map


def subject_stats_to_df(
    stats: SubjectStats, include_all_row: bool = True,
    subject_col: str = "subject", mean_col: str = "mean_acc", std_col: str = "std_acc"
) -> pd.DataFrame:
    df = pd.DataFrame(
        [(s, float(d.get("mean", np.nan)), float(d.get("std", np.nan)))
         for s, d in sorted(stats.items())],
        columns=[subject_col, mean_col, std_col]
    )
    if include_all_row and not df.empty:
        df.loc[len(df)] = ["__ALL__", float(np.nanmean(df[mean_col])), float(np.nanstd(df[mean_col]))]
    return df


def build_save_dir(dataset_result_root: str, space: str, mode: str, ctx: RunContext) -> str:
    space_dir = "channel_space" if space == "channel" else "parcel_space"
    parts = [dataset_result_root, space_dir]
    parts.append(mode)
    parts.extend([
        f"ft_{'-'.join(ctx.feature_types)}",
        f"clf_{ctx.clf_name}",
        f"prune_ch_sma_{ctx.prune_chans_sma}",
        f"sel_hrf_{ctx.g.sel_hrf_roi}",
    ])
    path = os.path.join(*parts)
    os.makedirs(path, exist_ok=True)
    return path


def save_results_df(df: pd.DataFrame, save_dir: str, filename: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    full_path = os.path.join(save_dir, filename)
    df.to_csv(full_path, index=False)
    print(f"Saved results to {full_path}")
    return full_path


def aggregate_runs(run_level_stats: List[SubjectStats]) -> SubjectStats:
    """
    Collapse a list of per-run SubjectStats into a single SubjectStats per subject.
    """
    from collections import defaultdict

    per_subject_means: Dict[str, List[float]] = defaultdict(list)

    # collect run means per subject
    for d in run_level_stats:
        for sid, stat in d.items():
            per_subject_means[sid].append(stat.get("mean", np.nan))

    # collapse to SubjectStats
    out: SubjectStats = {}
    for sid, means in per_subject_means.items():
        means_arr = np.array(means, dtype=float)
        out[sid] = {
            "mean": float(np.nanmean(means_arr)) if means_arr.size else np.nan,
            "std":  float(np.nanstd(means_arr))  if means_arr.size else np.nan,
        }
    return out


def min_significant_correct(n: int, p: float = 0.5, alpha: float = 0.05) -> int:
    for k in range(n + 1):
        if 1 - binom.cdf(k - 1, n, p) <= alpha:
            return k
    return n

# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def slope_feature(epochs: xr.DataArray) -> xr.DataArray:
    x = epochs['reltime'].values
    mean_x = x.mean()
    x_dev = x - mean_x
    denom = np.sum(x_dev ** 2)
    mean_y = epochs.mean('reltime')
    y_dev = epochs - mean_y
    numerator = (x_dev * y_dev).sum('reltime')
    features = numerator / denom
    return features


def _compute_features(
    epo: xr.DataArray,
    feature_types: List[str],
    ft_slices: Dict[str, slice],
    only_hbo: bool = True
) -> xr.DataArray:
    if isinstance(feature_types, str):
        feature_types = [feature_types]

    dim = 'channel' if 'channel' in epo.dims else 'parcel'
    ops = {
        "slope": slope_feature,
        "mean":  lambda x: x.mean("reltime"),
        "max":   lambda x: x.max("reltime"),
        "min":   lambda x: x.min("reltime"),
        "auc":   lambda x: (x * np.diff(x.reltime.values).mean()).sum("reltime"),
    }

    out = []
    for ft in feature_types:
        if ft not in ft_slices: raise ValueError(f"No time slice defined for feature '{ft}'")
        sliced = epo.sel(reltime=ft_slices[ft])
        key = next((k for k in ops if ft.lower().startswith(k)), None)
        if key is None: raise ValueError(f"Unknown feature_type: {ft}")
        f = ops[key](sliced).expand_dims(dim={"features": [ft]})
        out.append(f)

    features = xr.concat(out, dim="features")
    if 'chromo' in features.dims and only_hbo:
        features = features.sel(chromo='HbO')

    has_chromo = 'chromo' in features.dims
    stack_dims = ["chromo", dim, "features"] if has_chromo else [dim, "features"]
    return features.stack(flattened_features=stack_dims).pint.dequantify()


def extract_features(epo: xr.DataArray, ctx: RunContext,
                     long_chs: Optional[List[str]] = None,
                     prune_chs: Optional[List[str]] = None) -> np.ndarray:
    if long_chs:  epo = epo.sel(channel=[c for c in long_chs if c in epo.channel.values])
    if prune_chs: epo = epo.sel(channel=[ch for ch in prune_chs if ch in epo.channel.values])
    return _compute_features(epo, ctx.feature_types, ctx.ft_slices, only_hbo=ctx.g.only_hbo)


def map_to_img_space(epo: xr.DataArray, B: xr.DataArray, Adot: xr.DataArray,
                     clean_chs: List[str], parcels: List[str], prune=False) -> xr.DataArray:
    if prune: epo = epo.sel(channel=clean_chs)
    mask = np.isin(Adot.channel.values, clean_chs)
    B_clean = B[:, np.tile(mask, 2)] if prune else B
    B_sub = B_clean.sel(flat_vertex=np.isin(B.parcel, parcels))

    dC_brain, _ = fw.apply_inv_sensitivity(epo, B_sub)
    mapped = dC_brain.assign_coords({
        'parcel': ('vertex', B_sub.sel(chromo='HbO').coords['parcel'].values),
        'is_brain': ('vertex', B_sub.sel(chromo='HbO').coords['is_brain'].values)
    })
    return mapped


def extract_parcel_features(epo: xr.DataArray, ctx: RunContext,
                            B: xr.DataArray, Adot: xr.DataArray,
                            clean_chs: List[str], parcels: List[str], prune=False) -> xr.DataArray:
    epo_img = map_to_img_space(epo, B, Adot, clean_chs, parcels, prune=prune)
    epo_parcel = epo_img.groupby('parcel').mean()
    epo_parcel = epo_parcel.transpose('epoch', 'chromo', 'parcel', 'reltime')
    feats = _compute_features(epo_parcel, ctx.feature_types, ctx.ft_slices, only_hbo=ctx.g.only_hbo)
    return feats

# ──────────────────────────────────────────────────────────────────────────────
# Classification helpers
# ──────────────────────────────────────────────────────────────────────────────

def run_classifier(X, y, split_data, ctx: RunContext, k: Optional[int] = None) -> float:
    train_idx = split_data['train_indices']
    test_idx = split_data['test_indices']

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if ctx.reduce_features:
        kk = (k if k is not None else ctx.g.n_reduced_feat_ws)
        if X_train.shape[1] > kk:
            selector = SelectKBest(score_func=f_classif, k=kk)
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)

    ctx.classifier.fit(X_train, y_train)
    y_pred = ctx.classifier.predict(X_test)
    return accuracy_score(y_test, y_pred)


def _apply_hrf_roi(
    epo: xr.DataArray,
    split_data: Dict[str, Any],
    ctx: RunContext,
    prune_chs: Optional[List[str]]
) -> xr.DataArray:
    """Optional HRF-ROI channel selection based on GLM weights."""
    if (split_data.get('hrf_diff') is None) or (not ctx.g.sel_hrf_roi) or (epo.channel.values.size <= ctx.g.sel_hrf_roi):
        return epo

    hrf_diff = split_data['hrf_diff']
    # clip outliers
    hrf_diff = hrf_diff.where(np.abs(hrf_diff) < 1.0, 0)
    hrf_diff = hrf_diff.sel(channel=[ch for ch in hrf_diff.channel.values if ch in epo.channel.values])

    if prune_chs:
        hrf_diff = hrf_diff.sel(channel=[ch for ch in prune_chs if ch in hrf_diff.channel.values])

    sorted_indices = np.argsort(hrf_diff.values)[::-1]
    hrf_sorted = list(hrf_diff.channel[sorted_indices].values)
    n_sel = min(ctx.g.sel_hrf_roi, hrf_diff.size)
    # symmetric pick from both ends
    channel_roi = hrf_sorted[:n_sel // 2] + hrf_sorted[-n_sel // 2:]

    if len(set(channel_roi) & set(epo.channel.values)) > 1:
        epo = epo.sel(channel=[c for c in channel_roi if c in epo.channel.values])
    return epo


def _cv_loop(epo_data, splits, build_X, ctx: RunContext) -> List[float]:
    scores = []
    for i, split in enumerate(splits):
        y = LabelEncoder().fit_transform(split['y'])
        epo = epo_data if not isinstance(epo_data, list) else epo_data[i]
        X = build_X(epo, split)
        X = X.values if hasattr(X, "values") else np.asarray(X)
        scores.append(run_classifier(X, y, split, ctx, k=ctx.g.n_reduced_feat_ws))
    return scores


def within_subject_cv(run_data: Dict, ss_run_data: Optional[Dict], ctx: RunContext,
                      subject_id: str,
                      subset_channels: Optional[List[str]] = None,
                      prune_chs: Optional[List[str]] = None) -> SubjectStats:
    epo_data = run_data['epochs']
    splits   = (ss_run_data['splits'] if ss_run_data else run_data['splits'])

    def build_X(epo, split):
        if subset_channels:
            epo = epo.sel(channel=[c for c in subset_channels if c in epo.channel.values])
        epo = _apply_hrf_roi(epo, split, ctx, prune_chs)
        long_chs = ctx.ds.long_channels if ctx.dt == "long" else None
        return extract_features(epo, ctx, long_chs=long_chs, prune_chs=prune_chs)

    scores = _cv_loop(epo_data, splits, build_X, ctx)
    return {subject_id: {"mean": float(np.mean(scores)) if scores else np.nan,
                         "std":  float(np.std(scores))  if scores else np.nan}}


def parcel_within_subject_cv(run_data: Dict, ctx: RunContext,
                             Adot, B, clean_chs: List[str], parcels: List[str],
                             subject_id: str, prune: bool = False) -> SubjectStats:
    epo_data, splits = run_data['epochs'], run_data['splits']

    def build_X(epo, _split):
        return extract_parcel_features(epo, ctx, B, Adot, clean_chs, parcels, prune=prune)

    scores = _cv_loop(epo_data, splits, build_X, ctx)
    return {subject_id: {"mean": float(np.mean(scores)) if scores else np.nan,
                         "std":  float(np.std(scores))  if scores else np.nan}}


def loso_classification(all_data: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
                        subjects: List[str], ctx: RunContext,
                        k: Optional[int] = None) -> SubjectStats:
    X_all, y_all, subj_idx = [], [], []
    for i, s in enumerate(subjects):
        for X, y in all_data[s]:
            X_all.append(X); y_all.append(y); subj_idx += [i] * len(y)
    X_all, y_all, subj_idx = np.vstack(X_all), np.concatenate(y_all), np.asarray(subj_idx)
    results: SubjectStats = {}

    for test_i in np.unique(subj_idx):
        tr, te = subj_idx != test_i, subj_idx == test_i
        Xtr, Xte = X_all[tr], X_all[te]
        ytr, yte = y_all[tr], y_all[te]

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr) 
        Xte = scaler.transform(Xte)

        if ctx.reduce_features and Xtr.shape[1] > (k or ctx.g.n_reduced_feat_loso):
            kk = (k or ctx.g.n_reduced_feat_loso)
            sel = SelectKBest(score_func=f_classif, k=kk).fit(Xtr, ytr)
            Xtr, Xte = sel.transform(Xtr), sel.transform(Xte)

        ctx.classifier.fit(Xtr, ytr)
        acc = accuracy_score(yte, ctx.classifier.predict(Xte))
        results[subjects[test_i]] = {"mean": float(acc), "std": 0.0}
    return results


def get_parcel_loso_features(data, subjects, ctx: RunContext,
                             Adot, B, clean_ch_map, parcels, prune=False):
    all_data = {}
    for subject in subjects:
        all_data[subject] = []
        for run in data[subject]:
            epo_data = data[subject][run]['epochs']
            split_data = data[subject][run]['splits'][0]
            epo = epo_data if not isinstance(epo_data, list) else epo_data[0]
            clean_chs = clean_ch_map[subject][run]
            y = LabelEncoder().fit_transform(split_data['y'])
            X = extract_parcel_features(epo, ctx, B, Adot, clean_chs, parcels, prune=prune)
            X_np = X.values if hasattr(X, "values") else X
            all_data[subject].append((X_np, y))
    return all_data


def parcel_loso_classification(data, subjects, ctx: RunContext,
                               Adot, B, clean_ch_map, parcels, prune=False) -> SubjectStats:
    """
    LOSO (parcel).
    Returns {subject: {"mean": acc, "std": 0.0}}
    """
    all_data = get_parcel_loso_features(data, subjects, ctx, Adot, B, clean_ch_map, parcels, prune=prune)
    return loso_classification(all_data, subjects, ctx, k=ctx.g.n_reduced_feat_loso)
