import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os
import matplotlib.patheffects as pe


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def sp_map(sp):
    return f"{sp} cm"

int_map = {'01': '0.2 μM', '02': '0.4 μM', '03': '0.6 μM'}


def plot_grouped_bars_by_dt(data_dict, dt_labels_map, title, ylabel="Accuracy", ylim=(0.4, 1.0),
                            group_spacing=1.5, subject_spacing=1.0, figsize=(20, 8),
                            save=False, bar_width=0.6, std_dict=None, color_map=None, hatches_map=None):
    subjects_list = list(next(iter(data_dict.values())).keys())
    dt_list = list(data_dict.keys())

    if color_map is None:
        colors = plt.cm.tab10.colors
        color_map = {dt: colors[i % len(colors)] for i, dt in enumerate(dt_list)}

    positions, x_ticks, x_tick_labels, all_accuracies, bar_colors = [], [], [], [], []
    accuracies_by_dt = {dt: [] for dt in dt_list}

    x = 0
    for dt in dt_list:
        for subject in subjects_list:
            acc = data_dict[dt].get(subject, np.nan)
            positions.append(x)
            x_ticks.append(x)
            x_tick_labels.append(subject)
            all_accuracies.append(acc)
            accuracies_by_dt[dt].append(acc)
            bar_colors.append(color_map[dt])
            x += subject_spacing
        x += group_spacing

    plt.figure(figsize=figsize)
    errors = []
    for dt in dt_list:
        for subject in subjects_list:
            errors.append(std_dict[dt].get(subject, 0) if std_dict else 0)

    plt.bar(positions, all_accuracies, color=bar_colors, width=bar_width, yerr=errors, capsize=4)

    plt.xticks(x_ticks, x_tick_labels, rotation=45, fontsize=18)
    plt.ylabel(ylabel, fontsize=20)
    plt.ylim(*ylim)
    plt.title(title, fontsize=24)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.yticks(fontsize=16)

    legend_patches = []
    for dt in dt_list:
        avg_accuracy = np.nanmean(accuracies_by_dt[dt])
        legend_label = f"{dt_labels_map.get(dt, dt)} (avg: {avg_accuracy*100:.2f}%)"
        legend_patches.append(Patch(color=color_map[dt], label=legend_label))
    plt.legend(handles=legend_patches, fontsize=16,  loc="upper left")

    plt.tight_layout()
    plt.show()


def plot_grouped_bars_by_subject(data_dict, dt_labels_map, title, ylabel="Accuracy", ylim=(0.4, 1.0),
                                 bar_width=0.3, subject_spacing=1.5, figsize=(20, 8),
                                 save=True, std_dict=None, color_map=None, hatches_map=None):
    dt_list = list(data_dict.keys())
    subjects_list = sorted(list(next(iter(data_dict.values())).keys()))

    if color_map is None:
        colors = plt.cm.tab10.colors
        color_map = {dt: colors[i % len(colors)] for i, dt in enumerate(dt_list)}
    if hatches_map is None:
        hatches_map = {}

    x_ticks, x_tick_labels = [], []
    bar_positions = {dt: [] for dt in dt_list}
    accuracies = {dt: [] for dt in dt_list}

    for i, subject in enumerate(subjects_list):
        center = i * subject_spacing
        for j, dt in enumerate(dt_list):
            x_pos = center + j * bar_width
            bar_positions[dt].append(x_pos)
            accuracies[dt].append(data_dict[dt].get(subject, np.nan))
        x_ticks.append(center + (len(dt_list) - 1) * bar_width / 2)
        x_tick_labels.append(subject)

    plt.figure(figsize=figsize)
    for dt in dt_list:
        stds = [std_dict[dt].get(subject, 0) if std_dict else 0 for subject in subjects_list]
        plt.bar(bar_positions[dt], accuracies[dt], width=bar_width,
                color=color_map[dt], label=dt_labels_map.get(dt, dt),
                yerr=stds, capsize=4, hatch=hatches_map.get(dt, None))

    plt.xticks(x_ticks, x_tick_labels, rotation=45, fontsize=18)
    plt.ylabel(ylabel, fontsize=20)
    plt.ylim(*ylim)
    plt.title(title, fontsize=24)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.yticks(fontsize=16)

    legend_patches = []
    for dt in dt_list:
        avg_accuracy = np.nanmean(accuracies[dt])
        patch = Patch(
            facecolor=color_map[dt],
            hatch=hatches_map.get(dt, None),
            label=f"{dt_labels_map.get(dt, dt)} (avg: {avg_accuracy*100:.2f}%)",
            edgecolor='black'
        )
        legend_patches.append(patch)

    plt.legend(handles=legend_patches, fontsize=16, loc="upper left")
    plt.tight_layout()
    plt.show()


def barplot_subsets(avg_by_dt, std_by_dt, subset_keys, optodes_per_cm2, dt_labels, save_dir, spatial_scaling, int_scaling, save_plot):
    plt.figure(figsize=(7,5))
    dts = list(avg_by_dt.keys())
    w = 0.4 / max(1, len(dts))
    x = np.arange(len(subset_keys))
    for i, dt in enumerate(dts):
        plt.bar(x + (i - 0.5)*w, avg_by_dt[dt], width=w, yerr=std_by_dt[dt], capsize=3, label=dt_labels[dt])
    plt.xticks(x, [str(round(v, 2)) for v in optodes_per_cm2], fontsize=16)
    plt.xlabel("# Optodes / cm²", fontsize=18)
    plt.ylabel("Accuracy", fontsize=18)
    plt.ylim(0.4, 1.0)
    plt.grid(True)
    plt.legend(fontsize=16, loc="upper left")
    plt.title(f"Intensity {int_map[int_scaling]} Space {sp_map(spatial_scaling)}", fontsize=20)
    plt.tight_layout()
    if save_plot:
        base_name = f"sp_{spatial_scaling}_int_{int_scaling}"
        for ext in ("png","pdf","svg"):
            plt.savefig(os.path.join(save_dir, f"{base_name}.{ext}"))
    plt.show()


def raincloud_subsets_runs(
    run_scores,
    subset_keys,
    optodes_per_cm2,
    dt_labels,
    save_dir,
    spatial_scaling,
    int_scaling,
    save_plot=True,
    ylim=(0.4, 1.0)
):
    """
    Raincloud plot using *run-level* accuracies.

    Expected structure:
        run_scores[dt][sub_key][subject] = [run_acc_1, run_acc_2, ...]

    - x-axis: subsets (labeled by #optodes/cm²)
    - For each subset: one cloud per dt condition (e.g. no_ss / ss)
    - If there are exactly 2 dt conditions:
        -> true mirrored raincloud:
           two half-violins that share the same center (no gap),
           with run dots in their respective half.
    - If >2 dt conditions:
        -> compact full violins with small horizontal offsets.
    """

    os.makedirs(save_dir, exist_ok=True)
    dts = list(run_scores.keys())
    n_dts = len(dts)
    x_sub = np.arange(len(subset_keys), dtype=float)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color'][:n_dts]
    # visual tuning
    base_violin_width = 0.25      # half-width used for mirrored case
    dot_size = 26
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(9, 6))
    # Offsets for n_dts > 2 (small, to keep groups tight)
    if n_dts > 2:
        offsets = np.linspace(-0.22, 0.22, n_dts)
        violin_width = 0.16
        jitter_width = 0.05

    for j, sub_key in enumerate(subset_keys):
        center_x = x_sub[j]

        for i, dt in enumerate(dts):
            # ----- gather all run-level values for this subset & condition -----
            vals = []
            subj_dict = run_scores.get(dt, {}).get(sub_key, {})
            for _, run_list in subj_dict.items():
                for v in run_list:
                    if v is not None and not np.isnan(v):
                        vals.append(float(v))
            if not vals:
                continue
            vals = np.asarray(vals, dtype=float)
            # ----- position logic -----
            if n_dts == 2:
                # both share the same center; half-violins
                pos = [center_x]
                vwidth = base_violin_width
            else:
                # multiple conditions: compact side-by-side full violins
                pos = [center_x + offsets[i]]
                vwidth = violin_width
            # ----- violin ("cloud") -----
            parts = ax.violinplot(
                vals,
                positions=pos,
                widths=vwidth,
                showmeans=False,
                showextrema=False,
                showmedians=False,
            )
            body = parts["bodies"][0]
            body.set_facecolor(colors[i])
            body.set_edgecolor("none")
            body.set_alpha(0.35)
            if n_dts == 2:
                # make it half: dt 0 = left, dt 1 = right
                path = body.get_paths()[0]
                verts = path.vertices
                x_c = center_x

                if i == 0:
                    # keep left half
                    verts[verts[:, 0] > x_c, 0] = x_c
                else:
                    # keep right half
                    verts[verts[:, 0] < x_c, 0] = x_c
            # ----- dots ("rain"): one per run -----
            if n_dts == 2:
                # keep dots strictly in their half, tight to center
                if i == 0:
                    x_j = center_x + rng.uniform(-vwidth, 0.0, size=len(vals)) * 0.6
                else:
                    x_j = center_x + rng.uniform(0.0, vwidth, size=len(vals)) * 0.6
            else:
                x_j = pos[0] + rng.uniform(-jitter_width, jitter_width, size=len(vals))
            ax.scatter(
                x_j,
                vals,
                s=dot_size,
                color=colors[i],
                alpha=0.9,
                #edgecolor="black",
                linewidth=0.3,
                zorder=3,
            )
            # ----- mean line -----
            mean = np.mean(vals)
            if n_dts == 2:
                if i == 0:
                    x0 = center_x - vwidth * 0.6
                    x1 = center_x - vwidth * 0.1
                else:
                    x0 = center_x + vwidth * 0.1
                    x1 = center_x + vwidth * 0.6
            else:
                x0 = pos[0] - vwidth * 0.25
                x1 = pos[0] + vwidth * 0.25
            
            ax.plot([x0, x1], [mean, mean],
                    color=colors[i],
                    linewidth=2,
                    zorder=4,
                    path_effects=[
                        pe.Stroke(linewidth=4, foreground='black'),
                        pe.Normal()
                    ]
            )
    # ----- axes & legend -----
    ax.set_xticks(x_sub)
    ax.set_xticklabels([str(round(v, 2)) for v in optodes_per_cm2], fontsize=18)
    ax.set_xlabel("# Optodes / cm²", fontsize=20)
    ax.set_ylabel("Accuracy", fontsize=20)
    if ylim is not None:
        ax.set_ylim(*ylim)
    plt.yticks(fontsize=18)
    ax.grid(True, axis="y", alpha=0.3)
    handles = [
        Patch(facecolor=colors[i],
              edgecolor="black",
              alpha=0.5,
              label=dt_labels.get(dt, dt))
        for i, dt in enumerate(dts)
    ]
    leg = ax.legend(handles=handles, fontsize=18, loc="upper left", frameon=True)
    leg.get_title().set_fontweight('bold')
    #plt.title(f"Intensity {int_map[int_scaling]} Space {sp_map(spatial_scaling)}", fontsize=22, fontweight='bold')
    plt.tight_layout()
    if save_plot:
        base_name = f"sp_{spatial_scaling}_int_{int_scaling}"
        #for ext in ("png", "pdf", "svg"):
        for ext in ["pdf"]:
            fig.savefig(
                os.path.join(save_dir, f"{base_name}_raincloud.{ext}"),
                dpi=300
            )
    plt.show()


def loso_barplot(loso_stats, title, save_dir, fname, save_plot):
    plt.figure(figsize=(12,5))
    names = list(loso_stats.keys()) 
    vals = [loso_stats[s]["mean"] for s in names]
    plt.bar(names, vals)
    plt.ylabel("LOSO Accuracy")
    plt.xlabel("Left-out Subject")
    plt.title(title)
    plt.ylim(0.4,1)
    plt.axhline(0.5, linestyle='--', label='Chance')
    plt.axhline(np.mean(vals), linestyle='--', label=f'Mean ({np.mean(vals)*100:.2f}%)')
    plt.legend()
    plt.tight_layout()
    if save_plot: plt.savefig(os.path.join(save_dir, fname))
    plt.show()