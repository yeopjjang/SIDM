"""ABCD finalization helpers consolidated by responsibility.

The source sections retain the original algorithms and guards. Static, prefixed
global names isolate study-specific settings; namespaces expose the original
APIs without loading deleted modules or embedding executable source strings.
"""

from abcd_finalize_common import _FINALIZE_NAMESPACES, _opt_distribution_edges, _opt_fold_distribution
abcd_4mu_final_09style = _FINALIZE_NAMESPACES['abcd_4mu_final_09style']
abcd_4mu_final_09style_scans = _FINALIZE_NAMESPACES['abcd_4mu_final_09style_scans']
abcd_4mu_final_09style_validation = _FINALIZE_NAMESPACES['abcd_4mu_final_09style_validation']
abcd_capped_gap_plots = _FINALIZE_NAMESPACES['abcd_capped_gap_plots']
abcd_comprehensive_plots = _FINALIZE_NAMESPACES['abcd_comprehensive_plots']
abcd_data_expected_significance = _FINALIZE_NAMESPACES['abcd_data_expected_significance']
abcd_final_check_plots = _FINALIZE_NAMESPACES['abcd_final_check_plots']
abcd_final_comparison_plots = _FINALIZE_NAMESPACES['abcd_final_comparison_plots']
abcd_finalize_plots = _FINALIZE_NAMESPACES['abcd_finalize_plots']
abcd_iso_buffer_plots = _FINALIZE_NAMESPACES['abcd_iso_buffer_plots']
abcd_local_scan_plots = _FINALIZE_NAMESPACES['abcd_local_scan_plots']
abcd_sideband_plots = _FINALIZE_NAMESPACES['abcd_sideband_plots']


# ========================================================================
# abcd_4mu_final_09style: original study API
# ========================================================================
"""Notebook-09 presentation order and rendering adapters for the 4mu study.

The numerical MC and public-DATA results are consumed without changing their
selection, normalization, eligibility, or disclosure decisions.
"""
from pathlib import Path as finalize_4mu_final_09style_Path

import pandas as finalize_4mu_final_09style_pd


def finalize_4mu_final_09style_reference_presentation_order(notebook_dir):
    """Use the exact full and decision order already exported by notebook 09."""
    source = finalize_4mu_final_09style_Path(notebook_dir) / '09_2mu2e_final_check_outputs'
    full_manifest = finalize_4mu_final_09style_pd.read_csv(source / 'figure_manifest.csv')
    short_manifest = finalize_4mu_final_09style_pd.read_csv(source / 'decision_figure_manifest.csv')
    full = full_manifest.sort_values('order' if 'order' in full_manifest else 'pdf_page').name.tolist()
    short = short_manifest.sort_values('order' if 'order' in short_manifest else 'pdf_page').name.tolist()
    if len(full) != 45 or len(set(full)) != 45 or len(short) != 18 or not set(short).issubset(full):
        raise AssertionError('Notebook 09 must define 45 full-deck and 18 meeting-deck figures.')
    return full, short


def finalize_4mu_final_09style_make_09_style_figures(result):
    """Return the same 45 named presentation slots as notebook 09, with 4mu inputs."""
    make_scan_figures = _FINALIZE_NAMESPACES['abcd_4mu_final_09style_scans'].make_scan_figures
    make_validation_figures = _FINALIZE_NAMESPACES['abcd_4mu_final_09style_validation'].make_validation_figures

    full, short = finalize_4mu_final_09style_reference_presentation_order(result['config']['notebook_dir'])
    pieces = [make_scan_figures(result), make_validation_figures(result)]
    figures, captions, definitions, tables, metadata = {}, {}, [], {}, {}
    for label, piece in zip(('scans', 'validation'), pieces):
        for name, figure in piece['figures']:
            if name in figures:
                raise AssertionError('Duplicate notebook-09 presentation slot: ' + name)
            figures[name] = figure
        if set(captions).intersection(piece['captions']):
            raise AssertionError('Duplicate presentation caption.')
        captions.update(piece['captions'])
        definitions.append(piece['definitions'])
        for name, table in piece.get('tables', {}).items():
            if name in tables or name in result['tables']:
                raise AssertionError('A presentation adapter cannot replace a numerical table: ' + name)
            tables[name] = table
        metadata[label] = piece.get('metadata', {})
    if set(figures) != set(full) or set(captions) != set(full):
        raise AssertionError('All 45 notebook-09 slots require a figure and caption.')
    definition = finalize_4mu_final_09style_pd.concat(definitions, ignore_index=True)
    if definition.name.duplicated().any() or set(definition.name) != set(full):
        raise AssertionError('All 45 presentation definitions must be unique.')
    tables['figure_definitions'] = definition.set_index('name').loc[full].reset_index()
    metadata.update(reference_notebook='09_MuIso_DPhi_2mu2e_WP1_WP2_FinalCheck.ipynb',
        exact_reference_keys=True, full_deck_order=full, meeting_deck_order=short,
        numerical_study_changed=False, new_DATA_queries=0,
        comparison_markers_are_selected_methods=False)
    return dict(figures=[(name, figures[name]) for name in full], captions=captions,
                tables=tables, metadata=metadata)


# ========================================================================
# abcd_finalize_plots: original study API
# ========================================================================
"""Finalized VR figures preserving the reference 13–16 colors and panel layout.

Input tables supply all physics quantities. The plotting layer does not load
samples, select candidates, apply leakage criteria, or combine signal hypotheses.
"""
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm as LogNorm; from matplotlib.colors import Normalize as Normalize; from matplotlib.colors import SymLogNorm as SymLogNorm; from matplotlib.colors import TwoSlopeNorm as TwoSlopeNorm
from matplotlib.lines import Line2D as Line2D
from matplotlib.patches import Patch as Patch; from matplotlib.patches import Rectangle as Rectangle


_opt_ORDER = ()  # Set from config candidate_order by make_comparison_figures.
_opt_SAMPLES = (("MC", "VR"), ("DATA", "VR"))
_opt_COLORS = {"MC SR": "#0072B2", "MC VR": "#E69F00", "DATA VR": "#202020",
               "A": "#222222", "B": "#009E73", "C": "#D55E00", "D": "#CC79A7",
               "pred": "#D55E00"}


def _opt_frame(result, table, **filters):
    frame = result.get("tables", {}).get(table, pd.DataFrame())
    frame = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    for key, value in filters.items():
        if key not in frame:
            return frame.iloc[:0]
        frame = frame.loc[frame[key].eq(value)]
    return frame


def _opt_wp(result, key):
    selected = result["selected"]
    if not isinstance(selected, pd.DataFrame):
        selected = pd.DataFrame(selected)
    rows = selected.loc[selected["plane"].eq(key)]
    if len(rows) != 1:
        raise ValueError(f"Exactly one selected point is required for {key}.")
    return rows.iloc[0]


def _opt_angular(result, key):
    return True  # Every stored candidate uses Mu isolation × |Δφ|.


def _opt_title(result, key):
    spec, selected = result["specs"][key], _opt_wp(result, key)
    tx, ty = float(selected["tx"]), float(selected["ty"])
    relation = "≥" if _opt_angular(result, key) else "<"
    selection = spec.get("selection", "")
    if isinstance(selection, dict):
        selection = "; ".join(f"{name}={value}" for name, value in selection.items())
    elif isinstance(selection, (tuple, list)):
        selection = "; ".join(map(str, selection))
    selection = "\n".join(textwrap.wrap(str(selection), width=49))
    title = f"{spec.get('label', key)}\nx < {tx:g}; y {relation} {ty:g}"
    return title + ("\n" + selection if selection else "")


def _opt_channel(config):
    return config.get("topology_label", "2μ2e" if config.get("topology") == "2mu2e" else "4μ")


def _opt_figure(result, config, title, rows=1, sharey=False, height=None):
    height = height or (6.5 if rows == 1 else 8.5 if rows == 2 else 10.)
    columns = len(_opt_ORDER)
    fig, axes = plt.subplots(rows, columns, figsize=(max(8., 6. * columns), height),
                             squeeze=False, sharey=sharey)
    fig.suptitle(f"{_opt_channel(config)} · {title}", fontsize=12, y=.975)
    for index, key in enumerate(_opt_ORDER):
        axes[0, index].set_title(_opt_title(result, key), fontsize=11, pad=10)
    for ax in axes.ravel():
        ax.tick_params(labelsize=9)
        ax.grid(alpha=.25)
        ax.set_axisbelow(True)
    # Reserve independent bands for x labels, the shared legend and the caption.
    fig.subplots_adjust(left=min(.14, .255 / columns), right=.96, top=.79, bottom=.25,
                        wspace=.27, hspace=.35)
    return fig, axes


def _opt_caption(fig, caption):
    wrapped = '\n'.join(textwrap.fill(line, width=max(100, round(fig.get_figwidth() * 10.3))) for line in caption.splitlines())
    fig.text(.5, .012, wrapped, fontsize=8.5, va="bottom", ha="center")


def _opt_unavailable(ax, text="Unavailable", gate=False):
    ax.set_facecolor("#eeeeee" if gate else "#f7f7f7")
    ax.text(.5, .5, text, transform=ax.transAxes, ha="center", va="center", fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=.85))


def _opt_norm(arrays, closure=False):
    finite = [np.asarray(a)[np.isfinite(a)] for a in arrays]
    values = np.concatenate(finite) if finite else np.array([])
    if not values.size:
        return Normalize(0, 1), "viridis"
    if closure:
        return TwoSlopeNorm(vmin=min(float(values.min()), .9), vcenter=1.,
                            vmax=max(float(values.max()), 1.1)), "coolwarm"
    if np.any(values < 0):
        maximum = max(float(np.max(np.abs(values))), 1e-12)
        return SymLogNorm(linthresh=max(maximum * 1e-3, 1e-12), vmin=-maximum,
                          vmax=maximum), "coolwarm"
    positive = values[values > 0]
    if not positive.size:
        return Normalize(0, 1), "viridis"
    lower, upper = float(positive.min()), float(positive.max())
    if np.isclose(lower, upper):
        lower, upper = lower * .9, upper * 1.1
    return LogNorm(lower, upper), "viridis"


def _opt_errorbar(ax, x, y, error=None, **style):
    x, y = np.asarray(x), np.asarray(y, dtype=float)
    valid = np.isfinite(y)
    if not np.any(valid):
        return
    # Keep NaN gaps in the line, including DATA points failing the S/B criterion.
    ax.plot(x, np.where(valid, y, np.nan), **style)
    if error is not None:
        error = np.asarray(error, dtype=float)
        with_error = valid & np.isfinite(error) & (error >= 0)
        if np.any(with_error):
            ax.errorbar(x[with_error], y[with_error], yerr=error[with_error], capsize=2.5,
                        fmt="none", ecolor=style.get("color"), elinewidth=style.get("lw", 1.))


def _opt_dynamic_scale(axes, values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    nonzero = np.abs(finite[finite != 0])
    if not nonzero.size or not (np.any(finite < 0) or nonzero.max() / nonzero.min() > 100):
        return
    from matplotlib.ticker import SymmetricalLogLocator as SymmetricalLogLocator; from matplotlib.ticker import NullLocator as NullLocator
    threshold = max(float(np.percentile(nonzero, 10)) * .2, 1e-8)
    axes = np.asarray(axes).ravel()
    extents = np.array([[ax.dataLim.ymin, ax.dataLim.ymax] for ax in axes])
    extents = extents[np.isfinite(extents).all(axis=1)]
    for ax in axes:
        ax.set_yscale("symlog", linthresh=threshold)
        locator = SymmetricalLogLocator(linthresh=threshold, base=10)
        locator.set_params(numticks=7)
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_minor_locator(NullLocator())
    if len(extents):
        lower, upper = extents[:, 0].min(), extents[:, 1].max()
        transform = axes[0].yaxis.get_transform()
        limits = transform.transform([lower, upper])
        span = max(float(limits[1] - limits[0]), threshold)
        view = transform.inverted().transform([limits[0] - .08 * span, limits[1] + .12 * span])
        if lower >= 0:
            view[0] = max(0., view[0])
        for ax in axes:
            ax.set_ylim(*view)


def _opt_distribution_figures(result, config):
    """Keep the original common-bin colors, folding and ABCD quadrant placement."""
    figures = []
    for source, context in _opt_SAMPLES:
        planes = []
        for key in _opt_ORDER:
            plane = (result['data_planes'].get(key) if source == 'DATA'
                     else result['planes'][key][context])
            nominal = _opt_frame(result, 'closure', plane=key, source=source, context=context)
            if source == 'DATA' and (nominal.empty or not _opt_data_approved(nominal.iloc[0])):
                plane = None
            planes.append(plane)
        # The MC-derived display support also defines the per-bin DATA mask.
        display_planes = [result['planes'][key][context] for key in _opt_ORDER]
        xx = _opt_distribution_edges(display_planes, 'xedges', 2.)
        yy = _opt_distribution_edges(display_planes, 'yedges', np.pi)
        folded = [_opt_fold_distribution(plane, xx, yy, True)[0] if plane is not None else None
                  for plane in planes]
        if source == 'DATA':
            folded = [_opt_mask_distribution(array, result['distribution_purity'][key], xx, yy)
                      if array is not None else None for key, array in zip(_opt_ORDER, folded)]
        norm, cmap, color_label = _opt_distribution_norm([array for array in folded if array is not None])
        if source == 'DATA':
            color_label = color_label.replace('Weighted events', 'Events')
        columns = len(_opt_ORDER)
        fig, axes = plt.subplots(1, columns, figsize=(max(8., 16. * columns / 3.), 5.8), squeeze=False)
        axes = axes[0]
        mesh = None
        for ax, key, array in zip(axes, _opt_ORDER, folded):
            spec, selected = result['specs'][key], _opt_wp(result, key)
            ax.set_title(_opt_title(result, key), fontsize=10)
            ax.set_xlabel(spec['x_label']); ax.set_ylabel(spec['y_label'])
            ax.set(xlim=(xx[0], xx[-1]), ylim=(yy[0], yy[-1]), facecolor='#eeeeee')
            if array is None:
                _opt_unavailable(ax, 'DATA unavailable at this WP\nAll-hypothesis S/B criterion not passed', gate=True)
                continue
            mesh = ax.pcolormesh(xx, yy, array.T, norm=norm, cmap=cmap, shading='flat')
            if source == 'DATA':
                hidden = ~result['distribution_purity'][key]['approved']
                for ix, iy in np.argwhere(hidden):
                    ax.add_patch(Rectangle((xx[ix], yy[iy]), xx[ix + 1] - xx[ix], yy[iy + 1] - yy[iy],
                                           facecolor='#e6e6e6', edgecolor='#777777', hatch='///', lw=0., zorder=2))
                ax.text(.02, .98, f'Blinded bins: {int(hidden.sum())}/{hidden.size}',
                        transform=ax.transAxes, va='top', fontsize=8,
                        bbox=dict(facecolor='white', alpha=.85, edgecolor='none', pad=2))
            tx, ty = float(selected.tx), float(selected.ty)
            ax.axvline(tx, color='#cc3311', lw=1.7); ax.axhline(ty, color='#cc3311', lw=1.7)
            lowx, highx = (xx[0] + tx)/2, (tx + xx[-1])/2
            lowy, highy = (yy[0] + ty)/2, (ty + yy[-1])/2
            for name, (x, y) in zip('ABCD', ((lowx, highy), (highx, highy), (lowx, lowy), (highx, lowy))):
                ax.text(x, y, name, ha='center', va='center', color='white', fontsize=12,
                        bbox=dict(facecolor='black', alpha=.65, edgecolor='none', pad=2))
        detail = 'candidate distributions' if context == 'VR' else 'nominal SR distribution'
        fig.suptitle(f"{_opt_channel(config)} {source} {context} · Mu iso × |Δφ| {detail}", fontsize=13)
        distribution_note = ('DATA: fixed-WP ABCD gate plus binwise max(S/B_MC) < 0.10 over 120 hypotheses; hatched bins are blinded.\n'
                             'Invalid/nonpositive MC support is also blinded; only visible DATA bins set the color scale.'
                             if source == 'DATA' else 'MC distributions are displayed without the DATA purity mask.')
        fig.text(.5, .012, 'Display uses exact common bins to iso=2; higher-isolation tails and overflow fold into the final bin.\n'
                 'Isolation underflow is excluded; angle flows fold into edge bins. A/B pass the high-angle boundary.\n'
                 'Folding is display-only. ' + distribution_note,
                 ha='center', va='bottom', fontsize=8.5)
        fig.subplots_adjust(top=.70, bottom=.25, left=min(.12, .195 / columns), right=.86, wspace=.30)
        if mesh is not None:
            fig.colorbar(mesh, cax=fig.add_axes([.90, .27, .014, .42]), label=color_label)
        figures.append((fig, f'{source.lower()}_{context}_planes'))
    return figures


def _opt_closure_figure(result, config):
    """Use the 05–10 observed/predicted and closure style for each source/context."""
    figures = []
    for source, context in _opt_SAMPLES:
        columns = len(_opt_ORDER)
        fig, axes = plt.subplots(2, columns, figsize=(max(8., 5. * columns), 7), sharey=False, squeeze=False,
                                 gridspec_kw={"height_ratios": [2.1, 1]})
        for column, key in enumerate(_opt_ORDER):
            top, bottom = axes[:, column]
            top.set_title(_opt_title(result, key), fontsize=11)
            rows = _opt_frame(result, "closure", plane=key, source=source, context=context)
            if rows.empty:
                _opt_unavailable(top)
                _opt_unavailable(bottom)
                bottom.set_ylim(0, 2)
                continue
            row = rows.iloc[0]
            if source == "DATA" and not _opt_data_approved(row):
                _opt_unavailable(top, "All-hypothesis S/B criterion failed", gate=True)
                _opt_unavailable(bottom, "DATA closure not evaluated", gate=True)
                bottom.set_ylim(0, 2)
                continue
            _opt_reference_errorbar(top, [0], [row["A"]], error=[row["A_error"]],
                                    fmt="o", capsize=4, color="#222222", label=f"Observed {source} A")
            _opt_reference_errorbar(top, [1], [row["pred"]], error=[row["pred_error"]],
                                    fmt="s", capsize=4, color="#D55E00", label="ABCD prediction BC/D")
            top.set_xticks([0, 1], [f"Observed {source} A", "BC/D"])
            top.set_xlim(-.55, 1.55)
            top.axhline(0, color="#888888", lw=.7)
            top.grid(axis="y", alpha=.25)
            lower, upper = top.get_ylim()
            top.set_ylim(lower, upper + .4 * (upper - lower))
            top.text(.04, .95, f"A = {_opt_summary_number(row['A'])}\n"
                     f"BC/D = {_opt_summary_number(row['pred'])}\n"
                     f"min N_eff = {_opt_summary_number(row['neff_min'])}"
                     + ("\nLOW STAT" if bool(row.get('low_stat', False)) else ""),
                     transform=top.transAxes, va="top", fontsize=9,
                     bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none", "pad": 1})
            kappa, sigma = float(row["kappa"]), float(row["kappa_error"])
            _opt_reference_errorbar(bottom, [.5], [kappa], error=[sigma],
                                    fmt="o", capsize=4, color="#0072B2")
            bottom.axhline(1, color="#555555", lw=1, ls="--")
            bottom.set_xlim(-.55, 1.55)
            bottom.set_xticks([])
            bottom.grid(axis="y", alpha=.25)
            annotation = (f"κ = {_opt_summary_number(kappa)} ± {_opt_summary_number(sigma)}"
                          if np.isfinite(sigma) else
                          f"κ = {_opt_summary_number(kappa)}; uncertainty undefined")
            bottom.text(.04, .56, annotation, transform=bottom.transAxes, va="top", fontsize=9,
                        bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none", "pad": 1})
            _opt_closure_limits(bottom, [.5], [kappa], [sigma])
            if not np.isfinite(kappa):
                bottom.text(.5, .4, "Closure undefined", ha="center", transform=bottom.transAxes, fontsize=10)
        axes[0, 0].set_ylabel("Weighted events" if source == "MC" else "Events")
        axes[1, 0].set_ylabel("κ = A / (BC/D)")
        _opt_summary_finish(fig, config, source, context, "Closure at the stored working point",
                            "Yield axes autoscale independently by column; all κ axes span 0–2. ▲/▼ mark outside values; full errors remain in tables.\n"
                            "N_eff = (Σw)²/Σw² is a weighted-statistics diagnostic; small N_eff weakens closure inference.")
        for ax in axes.ravel():
            ax.tick_params(axis="y", labelleft=True)
        figures.append((fig, f"{source.lower()}_{context}_selected_closure"))
    return figures


def _opt_cell_edges(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        width = max(abs(values[0]) * .05, .025)
        return np.array([values[0] - width, values[0] + width])
    return np.r_[values[0] - (values[1] - values[0]) / 2,
                 (values[1:] + values[:-1]) / 2, values[-1] + (values[-1] - values[-2]) / 2]


def _opt_grid_figures(result, config):
    from matplotlib.colors import LinearSegmentedColormap as LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "optimized_blue_gray_red", ["#2166ac", "#bdbdbd", "#b2182b"], N=257)
    cmap.set_under(cmap(0.)); cmap.set_over(cmap(1.)); cmap.set_bad("#eeeeee")
    norm = TwoSlopeNorm(vmin=0., vcenter=1., vmax=2.)
    figures = []
    for source, context in _opt_SAMPLES:
        frames = [_opt_frame(result, "grid", plane=key, source=source, context=context) for key in _opt_ORDER]
        fig, axes = _opt_figure(result, config, f"{source} {context} · fixed boundary scan closure maps", height=9.)
        fig.subplots_adjust(right=.88, top=.82, bottom=.22, wspace=.32)
        mesh = None
        for ax, key, frame in zip(axes[0], _opt_ORDER, frames):
            ax.set_xlabel(result["specs"][key]["x_label"] + " boundary", fontsize=9)
            ax.set_ylabel(result["specs"][key]["y_label"] + "\nboundary", fontsize=9, labelpad=5)
            if not len(frame):
                _opt_unavailable(ax, "No grid points available")
                continue
            if frame.duplicated(["tx", "ty"]).any():
                raise ValueError("Closure grid contains duplicate WP coordinates.")
            xs = np.sort(frame.tx.dropna().unique().astype(float))
            ys = np.sort(frame.ty.dropna().unique().astype(float))
            zz = np.full((len(ys), len(xs)), np.nan)
            blocked, undefined, low_stat = [], [], []
            for _, row in frame.iterrows():
                ix = int(np.flatnonzero(np.isclose(xs, row.tx, rtol=0, atol=1e-10))[0])
                iy = int(np.flatnonzero(np.isclose(ys, row.ty, rtol=0, atol=1e-10))[0])
                allowed = source != "DATA" or _opt_data_approved(row)
                value = float(row.get("kappa", np.nan))
                if not allowed:
                    blocked.append((ix, iy))
                elif np.isfinite(value):
                    zz[iy, ix] = value
                    flag = row.get("low_stat", False)
                    if pd.notna(flag) and bool(flag):
                        low_stat.append((ix, iy))
                else:
                    undefined.append((ix, iy))
            mesh = ax.imshow(np.ma.masked_invalid(zz), origin="lower", aspect="auto",
                             interpolation="nearest", cmap=cmap, norm=norm)
            for iy, ix in np.ndindex(zz.shape):
                value = zz[iy, ix]
                if not np.isfinite(value):
                    continue
                color = cmap(norm(value))
                luminance = .299 * color[0] + .587 * color[1] + .114 * color[2]
                ax.text(ix, iy, f"{value:.2f}" if abs(value) < 100 else f"{value:.1e}",
                        ha="center", va="center", fontsize=7,
                        color="white" if luminance < .48 else "#222222", zorder=4)
                if .9 <= value <= 1.1:
                    ax.add_patch(Rectangle((ix - .49, iy - .49), .98, .98,
                                           facecolor="none", edgecolor="#009E73", lw=1.7, zorder=3))
            for ix, iy in blocked:
                ax.add_patch(Rectangle((ix - .5, iy - .5), 1., 1., facecolor="#eeeeee",
                                       edgecolor="#999999", hatch="///", lw=0, zorder=3))
            for ix, iy in undefined:
                ax.plot(ix, iy, "x", color="#666666", ms=5, zorder=4)
            for ix, iy in low_stat:
                ax.plot(ix + .32, iy + .32, "o", color="#222222", ms=2.8,
                        markeredgecolor="white", markeredgewidth=.4, zorder=6)
            selected = _opt_wp(result, key)
            ix = np.flatnonzero(np.isclose(xs, float(selected.tx), rtol=0, atol=1e-10))
            iy = np.flatnonzero(np.isclose(ys, float(selected.ty), rtol=0, atol=1e-10))
            if len(ix) != 1 or len(iy) != 1:
                raise ValueError("The selected WP must occur in the stored closure grid.")
            ax.add_patch(Rectangle((ix[0] - .42, iy[0] - .42), .84, .84,
                                   facecolor="none", edgecolor="black", lw=2, zorder=5))
            ax.set_xticks(np.arange(len(xs)), [f"{value:g}" for value in xs], rotation=60, fontsize=8)
            ax.set_yticks(np.arange(len(ys)), ["π" if np.isclose(value, np.pi) else f"{value:g}" for value in ys], fontsize=8)
            ax.set_xlim(-.5, len(xs) - .5); ax.set_ylim(-.5, len(ys) - .5)
            ax.set_facecolor("#eeeeee")
            ax.spines[:].set_visible(True)
            ax.grid(False)
        if mesh is not None:
            fig.colorbar(mesh, cax=fig.add_axes([.91, .22, .015, .59]),
                         label="Closure κ = A / (BC/D)", ticks=np.arange(0, 2.1, .5))
        legend = [Patch(facecolor="none", edgecolor="black", lw=2, label="Fixed WP"),
                  Patch(facecolor="none", edgecolor="#009E73", lw=1.7, label="0.9 ≤ central κ ≤ 1.1"),
                  Patch(facecolor="#eeeeee", edgecolor="#999999", hatch="///", label="DATA S/B criterion failed"),
                  Line2D([], [], color="grey", marker="x", ls="none", label="κ undefined"),
                  Line2D([], [], color="#222222", marker="o", ms=3, ls="none", label="Low-stat corner dot")]
        if context == 'SR':
            legend = [legend[index] for index in (0, 1, 3, 4)]
        fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .08),
                   ncol=min(3, len(_opt_ORDER) + 1), frameon=False, fontsize=8.5)
        _opt_caption(fig, "Equal-spaced display of native positive scan thresholds through isolation 2 and |Δφ| π; zero is excluded because its ABCD split is degenerate.\n"
                     "Blue κ ≤ 0, grey κ = 1, red κ ≥ 2. Labels show actual κ; green outlines use central values only.\n"
                     f"Hatching: DATA S/B criterion failed; ×: undefined κ; corner dot: min N_eff < {config.get('min_neff', 10.):g}. "
                     "Full errors and statistical-support flags remain in tables.")
        figures.append((fig, f"{source.lower()}_{context}_closure_grid"))
    return figures


def _opt_composition_figures(result, config):
    """Keep background composition and effective counts separate, as in 05–10."""
    process_order = ("QCD", "DY", "TTJets", "Diboson")
    process_colors = {"QCD": "#56B4E9", "DY": "#E69F00", "TTJets": "#009E73", "Diboson": "#CC79A7"}
    regions = ("A", "B", "C", "D")
    figures = []
    for context in ("VR",):
        fig, axes = plt.subplots(1, len(_opt_ORDER), figsize=(5. * len(_opt_ORDER), 5.9), sharey=True)
        for ax, key in zip(axes, _opt_ORDER):
            rows = _opt_frame(result, "composition", plane=key, context=context)
            positive, negative = np.zeros(4), np.zeros(4)
            for group in process_order:
                component = rows.loc[rows["component"].eq(group)].set_index("region")
                if not component.index.is_unique:
                    raise ValueError(f"Repeated region yield for {key}/{context}/{group}.")
                values = component["yield"].reindex(regions, fill_value=0).to_numpy(float)
                pos, neg = np.maximum(values, 0), np.minimum(values, 0)
                ax.bar(np.arange(4), pos, bottom=positive, color=process_colors[group], label=group, width=.65)
                ax.bar(np.arange(4), neg, bottom=negative, color=process_colors[group], width=.65)
                positive += pos
                negative += neg
            total = _opt_frame(result, "closure", plane=key, source="MC", context=context)
            if len(total):
                row = total.iloc[0]
                _opt_reference_errorbar(ax, np.arange(4), [row[r] for r in regions],
                                        error=[row[r + "_error"] for r in regions],
                                        fmt="o", color="black", capsize=3, markersize=3, label="Total ± stat.")
            ax.axhline(0, color="#888888", lw=.8)
            ax.set_xticks(np.arange(4), regions)
            ax.set_xlabel("ABCD region")
            ax.set_title(_opt_title(result, key), fontsize=11)
            ax.grid(axis="y", alpha=.25)
        axes[0].set_ylabel("Weighted events")
        handles, names = axes[0].get_legend_handles_labels()
        fig.legend(handles, names, loc="lower center", bbox_to_anchor=(.5, .09), ncol=5, frameon=False, fontsize=9)
        _opt_summary_finish(fig, config, "MC", context, "Background composition at the exclusive candidate VRs",
                            "Positive and negative contributions are stacked separately; the total marker is the algebraic sum.\n"
                            "Each column uses its stored candidate WP and event selection; all yields and statistical errors are retained in tables.",
                            bottom=.23, top=.76)
        figures.append((fig, f"mc_{context}_composition"))

        fig, axes = plt.subplots(1, len(_opt_ORDER), figsize=(5. * len(_opt_ORDER), 5.8), sharey=True)
        for ax, key in zip(axes, _opt_ORDER):
            ax.set_title(_opt_title(result, key), fontsize=11)
            rows = _opt_frame(result, "closure", plane=key, source="MC", context=context)
            if rows.empty:
                _opt_unavailable(ax)
                continue
            row = rows.iloc[0]
            neff = np.array([row["neff_" + region] for region in regions], dtype=float)
            bars = ax.bar(np.arange(4), np.where(np.isfinite(neff), neff, 0), width=.65, color="#0072B2", alpha=.75)
            for bar, value in zip(bars, neff):
                if np.isfinite(value):
                    ax.annotate(_opt_summary_number(value), (bar.get_x() + bar.get_width() / 2, value),
                                xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9)
                else:
                    ax.text(bar.get_x() + bar.get_width() / 2, .04, "undefined", ha="center", va="bottom",
                            rotation=90, transform=ax.get_xaxis_transform(), fontsize=8)
            ax.set_xticks(np.arange(4), regions)
            ax.set_xlabel("ABCD region")
            ax.grid(axis="y", alpha=.25)
            minimum = config.get("min_neff")
            if minimum is not None:
                ax.axhline(minimum, color="#cc3311", lw=1, ls="--", label=f"Diagnostic reference: {minimum:g}")
            ax.margins(y=.18)
        axes[0].set_ylabel("N_eff = (Σw)² / Σw²")
        _opt_summary_finish(fig, config, "MC", context, "Effective background counts at the exclusive candidate VRs",
                            "Effective counts use stored weighted yields and variances. They are statistical support diagnostics, not raw event counts.\n"
                            "Small effective counts require caution when interpreting delta-method closure errors.", bottom=.17, top=.76)
        figures.append((fig, f"mc_{context}_effective_counts"))
    return figures


def _opt_correlation_figure(result, config):
    """Reuse the 05–10 correlation panels with the exclusive candidate VRs."""
    figures = []
    regions = ("All", "A", "B", "C", "D")
    for source, context in _opt_SAMPLES:
        table = _opt_frame(result, "correlations", source=source, context=context)
        fig, axes = plt.subplots(1, len(_opt_ORDER), figsize=(6. * len(_opt_ORDER), 5.5), sharex=True, sharey=True)
        mesh = None
        for ax, key in zip(axes, _opt_ORDER):
            rows = table.loc[table.plane.eq(key)] if len(table) else table
            if source == "DATA":
                nominal = _opt_frame(result, "closure", plane=key, source=source, context=context)
                if nominal.empty or not _opt_data_approved(nominal.iloc[0]):
                    rows = rows.iloc[:0]
                elif "approved" in rows:
                    rows = rows.loc[rows.approved.fillna(False).astype(bool)]
            ax.set_title(_opt_title(result, key), fontsize=11)
            ax.set_xticks(range(5), regions)
            ax.set_yticks([])
            ax.set_xlabel("ABCD window at the fixed WP")
            if rows.empty:
                _opt_unavailable(ax, "No DATA correlation passing the S/B criterion" if source == "DATA"
                                 else "Correlation unavailable", gate=source == "DATA")
                ax.set(xlim=(-.5, 4.5), ylim=(-.5, .5))
                continue
            if not rows.region.is_unique:
                raise ValueError("Correlation summary must contain one row per plane/source/context/region.")
            rows = rows.set_index("region").reindex(regions)
            rho = rows["rho"].to_numpy(float)
            cmap = plt.get_cmap("coolwarm").copy()
            cmap.set_bad("#eeeeee")
            mesh = ax.imshow(np.ma.masked_invalid(rho[None, :]), aspect="auto", origin="lower",
                             interpolation="nearest", norm=Normalize(-1, 1), cmap=cmap)
            for index, (_, row) in enumerate(rows.iterrows()):
                value, neff = float(row["rho"]), float(row["neff"])
                label = f"ρ = {value:.3f}" if np.isfinite(value) else "ρ undefined"
                neff_label = f"{neff:.3g}".replace("e+0", "e").replace("e-0", "e-") if np.isfinite(neff) else "undefined"
                label += "\nN_eff = " + neff_label
                if "LOW_STAT" in str(row.get("status", "")):
                    label += "\nLow statistics"
                ax.text(index, 0, label, ha="center", va="center", fontsize=8,
                        color="white" if np.isfinite(value) and abs(value) > .65 else "#222222")
        selection = "candidate VR comparison"
        mass = float(config.get("mass_min", 150.))
        fig.suptitle(f"{_opt_channel(config)} {source} · {context}: {selection}\n"
                     "Correlation and effective statistical support at the fixed WP", fontsize=13)
        _opt_caption(fig, "Pearson ρ uses stored native regular-bin support; underflow, overflow and display-only tail folding are excluded.\n"
                     "Negative bins, empty support or zero spread make ρ undefined. Low statistics marks the stored LOW_STAT status.\n"
                     + ("Only DATA candidates passing the S/B criterion are shown; full support and status remain in the numerical table."
                        if source == "DATA" else "The candidates share Mu iso × |Δφ| axes, with the event cuts shown above. Full support and status remain in the numerical table."))
        fig.subplots_adjust(left=.06, right=.90, top=.72, bottom=.24, wspace=.10)
        if mesh is not None:
            cax = fig.add_axes([.92, .26, .014, .44])
            fig.colorbar(mesh, cax=cax, label="Pearson ρ")
        figures.append((fig, f"{source.lower()}_{context}_correlations"))
    return figures


def _opt_closure_limits(ax, x, kappa, error, *, note_y=.87):
    """Reference 01/02 range and edge markers, with explicit omission counts."""
    x, kappa, error = np.asarray(x, float), np.asarray(kappa, float), np.asarray(error, float)
    finite = np.isfinite(kappa)
    finite_error = finite & np.isfinite(error) & (error >= 0)
    low, high = 0., 2.
    ax.set_ylim(low, high)
    below, above = finite & (kappa < low), finite & (kappa > high)
    for mask, marker, height in ((below, "v", .06), (above, "^", 1.94)):
        if np.any(mask):
            ax.scatter(x[mask], np.full(mask.sum(), height), marker=marker, s=26,
                       color="black", zorder=5)
    clipped = finite_error & ((kappa - error < low) | (kappa + error > high))
    notes = []
    if np.any(below | above):
        notes.append(f"▲/▼: {int((below | above).sum())} κ values outside 0–2")
    if np.any(clipped):
        notes.append(f"{int(clipped.sum())} error ranges clipped")
    if notes:
        ax.text(.02, note_y, "\n".join(notes), va="top", fontsize=7.5, color="#8a4b00",
                transform=ax.transAxes, bbox={"facecolor": "white", "alpha": .82, "edgecolor": "none", "pad": 1})


def _opt_separate_y_ticks(fig):
    """Keep logarithmic labels separated without changing any plotted data."""
    from matplotlib.ticker import FixedLocator as FixedLocator
    fig.canvas.draw()
    processed = set()
    for ax in fig.axes:
        if ax.get_yscale() != "symlog" or id(ax.yaxis.get_major_locator()) in processed:
            continue
        original = ax.yaxis.get_major_locator()
        processed.add(id(original))
        ticks = np.asarray(ax.get_yticks())
        low, high = sorted(ax.get_ylim())
        ticks = ticks[(ticks >= low) & (ticks <= high)]
        if not len(ticks):
            continue
        positions = ax.transData.transform(np.c_[np.zeros(len(ticks)), ticks])[:, 1]
        # Prefer zero when central log ticks crowd it; retain all well-separated ticks.
        priority = sorted(range(len(ticks)), key=lambda i: (ticks[i] != 0, abs(ticks[i])))
        keep = []
        for index in priority:
            if all(abs(positions[index]-positions[other]) >= 18 for other in keep):
                keep.append(index)
        if len(keep) < len(ticks):
            ax.yaxis.set_major_locator(FixedLocator(ticks[sorted(keep)]))


def _opt_valid_error(values, variances=None, errors=None):
    values = np.asarray(values, dtype=float)
    if errors is None:
        variances = np.asarray(variances, dtype=float)
        errors = np.sqrt(np.where(variances >= 0, variances, np.nan))
    errors = np.asarray(errors, dtype=float)
    good = np.isfinite(values) & np.isfinite(errors) & (errors >= 0)
    return values, errors, good


def _opt_reference_errorbar(ax, x, y, *, variance=None, error=None, **kwargs):
    x = np.asarray(x, dtype=float)
    y, err, valid = _opt_valid_error(y, variances=variance, errors=error)
    if np.any(valid):
        # Keep NaN gaps in curves so blocked/undefined neighboring points are
        # never connected as if an unqueried point had been filled in.
        ax.errorbar(x, np.where(valid, y, np.nan), yerr=np.where(valid, err, np.nan), **kwargs)
    # An undefined uncertainty must not become a zero-error measurement.
    values_only = np.isfinite(y) & ~valid
    if np.any(values_only):
        ax.scatter(x[values_only], y[values_only], marker="x", color=kwargs.get("color", "black"), s=25)
    return valid


def _opt_data_missing_markers(ax, threshold, allowed, kappa, rows):
    """Flag missing DATA points below the κ frame without assigning any value."""
    blocked = ~np.asarray(allowed, bool)
    undefined = np.asarray(allowed, bool) & ~np.isfinite(kappa)
    unavailable = np.zeros(len(rows), dtype=bool)
    if "status" in rows:
        unavailable = np.asarray(allowed, bool) & rows["status"].astype(str).str.contains(
            "DATA_NOT_LOADED|DATA_UNAVAILABLE", regex=True).to_numpy()
        undefined &= ~unavailable
    items = ((blocked, "x", "#777777", "S/B criterion failed"),
             (undefined, "D", "#444444", "κ undefined"),
             (unavailable, "s", "#b86b00", "DATA unavailable"))
    for mask, marker, color, label in items:
        if mask.any():
            style = {"color": color} if marker == "x" else {"facecolors": "none", "edgecolors": color}
            ax.scatter(np.asarray(threshold)[mask], np.full(mask.sum(), -.065),
                       transform=ax.get_xaxis_transform(), marker=marker, s=20,
                       linewidths=1, clip_on=False, zorder=7, **style)
    notes = [f"{label}: {int(mask.sum())}" for mask, _, _, label in items if mask.any()]
    if notes:
        ax.text(.98, .87, "\n".join(notes), ha="right", va="top", fontsize=7.5,
                transform=ax.transAxes, color="#555555",
                bbox={"facecolor": "white", "alpha": .82, "edgecolor": "none", "pad": 1})


def _opt_summary_number(value):
    return f"{float(value):.3g}" if np.isfinite(value) else "undefined"


def _opt_summary_finish(fig, config, source, context, detail, caption, *, bottom=.15, top=.78):
    region = "candidate VR comparison" if context == "VR" else "nominal displaced–displaced reference"
    fig.suptitle(f"{_opt_channel(config)} {source} · {context}: {region}\n{detail}", fontsize=13)
    fig.text(.5, .012, caption, ha="center", va="bottom", fontsize=9, wrap=True)
    fig.subplots_adjust(top=top, bottom=bottom, left=min(.13, .24 / len(_opt_ORDER)),
                        right=.97, wspace=.08, hspace=.20)






def _opt_distribution_norm(arrays):
    """Preserve the original shared log/signed-symlog color convention."""
    finite = np.concatenate([array[np.isfinite(array)].ravel() for array in arrays]) if arrays else np.array([])
    if not finite.size or np.all(finite == 0):
        return Normalize(vmin=0, vmax=1), "viridis", "Weighted events / common bin"
    if np.any(finite < 0):
        vmax = max(float(np.max(np.abs(finite))), 1e-12)
        return (SymLogNorm(linthresh=max(vmax * 1e-3, 1e-12), vmin=-vmax, vmax=vmax),
                "coolwarm", "Weighted events / common bin (signed symlog)")
    positive = finite[finite > 0]
    vmin, vmax = float(positive.min()), float(positive.max())
    if np.isclose(vmin, vmax):
        vmin, vmax = vmin * .9, vmax * 1.1
    return LogNorm(vmin=vmin, vmax=vmax), "viridis", "Weighted events / common bin (log color)"


def _candidate_purity_limits(values, errors, limit):
    values, errors = np.asarray(values, float), np.asarray(errors, float)
    supported = np.isfinite(values) & (values >= 0)
    upper = values[supported] + np.where(np.isfinite(errors[supported]), errors[supported], 0.)
    maximum = max(float(np.max(upper)) if len(upper) else 0., 1e-8)
    high = max(maximum * 1.18, float(limit) * 1.12)
    return 0., high


def _candidate_purity_axis(ax, limits, config):
    """Always include the ten-percent reference and every finite diagnostic."""
    from matplotlib.ticker import ScalarFormatter as ScalarFormatter
    threshold = float(config.get("purity_limit", .1))
    ax.set_ylim(*limits)
    ax.tick_params(axis="y", labelcolor="tab:purple", labelleft=True, labelsize=8)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 3))
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.get_offset_text().set_color("tab:purple")
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.axhline(threshold, color="#666666", ls=":", lw=1.1)


def _candidate_scan_diagnostics(result, key, direction, thresholds, context="VR"):
    """Use one MC table for both MC and DATA diagnostic panels, without a mask."""
    rows = _opt_frame(result, "scan", plane=key, source="MC", context=context,
                      scan_axis=direction).sort_values("threshold")
    if rows.threshold.duplicated().any():
        raise ValueError(f"Repeated MC diagnostic threshold for {key}/{direction}.")
    return rows.set_index("threshold").reindex(thresholds)


def _opt_scan_figures(result, config):
    """Three rows: observed/predicted yield, closure, all-hypothesis worst S/B."""
    figures = []
    limits = {}
    for context in dict.fromkeys(context for _, context in _opt_SAMPLES):
        for direction in ("x", "y"):
            diagnostics = _opt_frame(result, "scan", source="MC", context=context, scan_axis=direction)
            limits[context, direction] = _candidate_purity_limits(diagnostics.max_S_B, diagnostics.max_S_B_error,
                                                                 config.get("purity_limit", .1))
    for source, context in _opt_SAMPLES:
        for direction in ("x", "y"):
            columns = len(_opt_ORDER)
            fig, axes = plt.subplots(3, columns, figsize=(max(8., 16. * columns / 3.), 10.8),
                                     sharex="col", sharey=False, squeeze=False,
                                     gridspec_kw={"height_ratios": [1.8, 1.3, 1.4]})
            for column, key in enumerate(_opt_ORDER):
                spec, selected = result["specs"][key], _opt_wp(result, key)
                rows = _opt_frame(result, "scan", plane=key, source=source, context=context,
                                  scan_axis=direction).sort_values("threshold")
                other = "ty" if direction == "x" else "tx"
                fixed_label = spec[("y" if direction == "x" else "x") + "_label"]
                fixed = float(selected[other])
                selection = "\n".join(textwrap.wrap(str(spec["selection"]), width=52))
                axes[0, column].set_title(f"{spec['label']}\n{selection}\n"
                                          f"{fixed_label} boundary = {fixed:g} (fixed)", fontsize=9)
                axes[2, column].set_xlabel(spec[direction + "_label"] + " boundary\n"
                                           + ("(high side passes)" if direction == "y" else "(low side passes)"))
                scan_name = "scan_mu_iso" if direction == "x" else "scan_abs_dphi"
                defaults = ([.05, .1, .15, .2, .25, .3, .4, .5, .75, 1., 2.] if direction == "x"
                            else [.5, 1., 1.5, 2., 2.2, 2.4, 2.6, 2.7, 2.8, 2.9, 3., 3.05, 3.1, np.pi])
                scan_ticks = np.asarray(spec.get(scan_name, config.get(scan_name, defaults)), dtype=float)
                major_ticks = [0., .5, 1., 1.5, 2.] if direction == "x" else [0., .5, 1., 1.5, 2., 2.5, np.pi]
                for ax in axes[:, column]:
                    ax.set_xlim(0, 2. if direction == "x" else np.pi)
                    ax.set_xticks(major_ticks, ["π" if np.isclose(value, np.pi) else f"{value:g}"
                                               for value in major_ticks])
                    ax.set_xticks(scan_ticks, minor=True)
                band = config.get("closure_band", (.9, 1.1))
                axes[1, column].axhspan(*band, color="#009E73", alpha=.12, zorder=0)
                for boundary in band:
                    axes[1, column].axhline(boundary, color="#009E73", lw=.9, ls=":")
                if rows.empty:
                    for ax in axes[:, column]:
                        _opt_unavailable(ax, "No scan points")
                    axes[1, column].set_ylim(0, 2)
                    continue
                threshold = rows.threshold.to_numpy(float)
                if len(np.unique(threshold)) != len(threshold) or not np.isfinite(threshold).all():
                    raise ValueError("Each candidate scan requires unique finite thresholds.")
                if not np.all(np.isclose(rows[other], fixed, rtol=0, atol=1e-10)):
                    raise ValueError("The opposite scan boundary must match the candidate WP.")
                allowed = (rows.approved.fillna(False).to_numpy(bool) if source == "DATA"
                           else np.ones(len(rows), dtype=bool))
                def shown(name):
                    return np.where(allowed, rows[name].to_numpy(float), np.nan)
                _opt_reference_errorbar(axes[0, column], threshold, shown("A"), error=shown("A_error"),
                                        fmt="o-", markersize=4, lw=1, capsize=3, color="#222222",
                                        label=f"Observed {source} A")
                _opt_reference_errorbar(axes[0, column], threshold, shown("pred"), error=shown("pred_error"),
                                        fmt="s--", markersize=4, lw=1, capsize=3, color="#D55E00", label="BC/D")
                kappa, error = shown("kappa"), shown("kappa_error")
                _opt_reference_errorbar(axes[1, column], threshold, kappa, error=error,
                                        fmt="o-", markersize=4, lw=1, capsize=3, color="#0072B2")
                _opt_closure_limits(axes[1, column], threshold, kappa, error)
                axes[0, column].axhline(0, color="#888888", lw=.7)
                axes[1, column].axhline(1, color="#555555", lw=1, ls="--")
                diagnostics = _candidate_scan_diagnostics(result, key, direction, threshold, context)
                values, errors = diagnostics.max_S_B.to_numpy(float), diagnostics.max_S_B_error.to_numpy(float)
                _opt_reference_errorbar(axes[2, column], threshold, values, error=errors,
                                        fmt="o-", color="tab:purple", markersize=4, lw=1.3, capsize=3)
                failed = ((~diagnostics.approved.fillna(False).to_numpy(bool) & np.isfinite(values))
                          if context == "VR" else np.zeros(len(values), dtype=bool))
                if failed.any():
                    axes[2, column].scatter(threshold[failed], values[failed], marker="s", s=42,
                                            facecolors="none", edgecolors="#b2182b", zorder=5)
                unsupported = ~np.isfinite(values)
                if unsupported.any():
                    axes[2, column].text(.02, .06, f"Undefined diagnostic: {int(unsupported.sum())}",
                                            transform=axes[2, column].transAxes, fontsize=8, color="#666666")
                _candidate_purity_axis(axes[2, column], limits[context, direction], config)
                for ax in axes[:, column]:
                    ax.axvline(float(selected["t" + direction]), color="#cc3311", lw=1, ls=":")
                    ax.grid(alpha=.25)
                    ax.tick_params(axis="y", labelleft=True, labelsize=8)
                handles, labels = axes[0, column].get_legend_handles_labels()
                if handles:
                    axes[0, column].legend(handles, labels, fontsize=8, loc="best")
                if source == "DATA":
                    _opt_data_missing_markers(axes[1, column], threshold, allowed, kappa, rows)
            axes[0, 0].set_ylabel("Weighted events in A" if source == "MC" else "Events in A")
            axes[1, 0].set_ylabel("κ = A / (BC/D)")
            axes[2, 0].set_ylabel("Worst MC S/B\nmax over 120 hypotheses × A/B/C/D", color="tab:purple")
            scan_label = "candidate boundary scan" if context == "VR" else "nominal SR boundary scan"
            fig.suptitle(f"{_opt_channel(config)} · {source} {context} · {scan_label} ({direction})", fontsize=13)
            caption = ("All native positive boundaries through isolation 2 and |Δφ| π are scanned; zero is excluded because its ABCD split is degenerate.\n"
                       "Top axes autoscale independently; κ axes span 0–2 with the green 0.9–1.1 band. Other boundaries and event cuts stay fixed.\n"
                       "Bottom panels use the same unmasked MC diagnostic and scale in MC/DATA: max S/B across 120 hypotheses and A/B/C/D.\n"
                       "Errors are MC statistical errors for the maximizing hypothesis/region; the maximizing identity can change between points.\n"
                       "The 10% S/B dotted line is always shown; red open squares mark failed VR criteria. No leakage veto is used.\n"
                       + ("DATA closure is shown only where the all-hypothesis S/B criterion passes; flags below κ mark unavailable or undefined points."
                          if source == "DATA" else "MC closure retains every point. Signal hypotheses are never summed; the diagnostic also retains purity failures."))
            if context == "SR":
                caption = ("All native positive boundaries through isolation 2 and |Δφ| π are scanned; the other boundary stays fixed.\n"
                           "Axes start at zero; the zero boundary is excluded because its ABCD split is degenerate.\n"
                           "κ axes span 0–2; green shading and dotted lines mark 0.9–1.1.\n"
                           "Bottom: max MC S/B over 120 hypotheses × A/B/C/D, with the maximizing pair's MC statistical error.\n"
                           "The 10% line is a visual reference only in SR. No SR values are masked by the VR criterion.\n"
                           "Signal hypotheses are not summed; the maximizing identity and undefined points remain in the numerical tables.")
            fig.text(.5, .012, caption, ha="center", va="bottom", fontsize=8.5)
            fig.subplots_adjust(top=.81, bottom=.235, left=min(.15, .225 / columns),
                                right=.96, wspace=.28, hspace=.16)
            figures.append((fig, f"{source.lower()}_{context}_{direction}_scan"))
    return figures


def _candidate_purity_figure(result, config):
    rows = _opt_frame(result, "purity_regions")
    limits = _candidate_purity_limits(rows.max_S_B, rows.max_S_B_error, config.get("purity_limit", .1))
    fig, axes = plt.subplots(1, len(_opt_ORDER), figsize=(16. * len(_opt_ORDER) / 3., 6), sharey=False)
    for ax, key in zip(axes, _opt_ORDER):
        part = rows.loc[rows.plane.eq(key)].set_index("region")
        if not part.index.is_unique:
            raise ValueError("Expected one nominal S/B maximum per candidate and ABCD region.")
        part = part.reindex(list("ABCD"))
        values, errors = part.max_S_B.to_numpy(float), part.max_S_B_error.to_numpy(float)
        ax.bar(np.arange(4), np.where(np.isfinite(values), values, 0.), color="tab:purple", width=.6, alpha=.65)
        _opt_reference_errorbar(ax, np.arange(4), values, error=errors, fmt="o", color="tab:purple", capsize=4, ms=4)
        failed = ~part.approved.fillna(False).to_numpy(bool) & np.isfinite(values)
        if failed.any():
            ax.scatter(np.arange(4)[failed], values[failed], marker="s", s=55,
                       facecolors="none", edgecolors="#b2182b", zorder=5)
        for index, value in enumerate(values):
            if not np.isfinite(value):
                ax.text(index, .03, "undefined", transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
        _candidate_purity_axis(ax, limits, config)
        ax.set_xticks(range(4), list("ABCD")); ax.set_xlabel("ABCD region at the candidate WP")
        ax.set_title(_opt_title(result, key), fontsize=10)
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Max MC S/B across 120 hypotheses", color="tab:purple")
    fig.suptitle(f"{_opt_channel(config)} · nominal all-hypothesis signal/background diagnostic", fontsize=13)
    fig.text(.5, .012, "Each bar independently maximizes S/B over 120 hypotheses in that ABCD region; hypotheses are never summed.\n"
             "Errors describe the maximizing hypothesis's MC statistical uncertainty. Worst sample identities remain in purity_regions.csv.\n"
             "The 10% criterion is always shown; red open squares mark failures. Low effective MC statistics remain a limitation.",
             ha="center", va="bottom", fontsize=8.5)
    fig.subplots_adjust(top=.70, bottom=.24, left=.08, right=.97, wspace=.27)
    return [(fig, "mc_VR_purity_regions")]


def _candidate_combined_closure(result, config):
    fig, axes = plt.subplots(1, len(_opt_ORDER), figsize=(16. * len(_opt_ORDER) / 3., 6.3), sharey=True)
    for ax, key in zip(axes, _opt_ORDER):
        ax.set_title(_opt_title(result, key), fontsize=10)
        notes = []
        for index, (source, color, marker) in enumerate((("MC", "#0072B2", "s"), ("DATA", "#222222", "o"))):
            rows = _opt_frame(result, "closure", plane=key, source=source, context="VR")
            if rows.empty:
                continue
            row = rows.iloc[0]
            approved = source == "MC" or _opt_data_approved(row)
            if not approved:
                notes.append(f"{source}: S/B criterion failed")
                continue
            kappa, error = float(row.kappa), float(row.kappa_error)
            _opt_reference_errorbar(ax, [index], [kappa], error=[error], fmt=marker, color=color, capsize=5, ms=6)
            low_stat = " · LOW STAT" if bool(row.get("low_stat", False)) else ""
            notes.append(f"{source}: κ={_opt_summary_number(kappa)} ± {_opt_summary_number(error)}"
                         f"; min N_eff={_opt_summary_number(row.neff_min)}{low_stat}")
            if np.isfinite(kappa) and not 0 <= kappa <= 2:
                ax.scatter([index], [1.92 if kappa > 2 else .08], marker="^" if kappa > 2 else "v", color=color, s=30)
        ax.axhspan(*config.get("closure_band", (.9, 1.1)), color="#009E73", alpha=.1)
        ax.axhline(1, color="#555555", ls="--", lw=1)
        ax.set_xticks([0, 1], ["MC", "DATA"]); ax.set_xlim(-.5, 1.5); ax.set_ylim(0, 2)
        ax.tick_params(axis="y", labelleft=True); ax.grid(axis="y", alpha=.25)
        ax.text(.03, .03, "\n".join(notes), transform=ax.transAxes, va="bottom", fontsize=7.7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
    axes[0].set_ylabel("κ = A / (BC/D)")
    fig.suptitle(f"{_opt_channel(config)} · fixed-WP closure: MC and DATA", fontsize=13)
    fig.text(.5, .012, "Green shading marks the central-value reference 0.9–1.1; it is not an uncertainty or a goodness-of-fit decision.\n"
             "All panels span κ=0–2; arrows mark outside values. Numerical errors and low-statistics flags remain in closure.csv.\n"
             "Displacement categories are mutually exclusive; working points within each scan are correlated. Good DATA closure does not imply good MC closure.",
             ha="center", va="bottom", fontsize=8.5)
    fig.subplots_adjust(top=.72, bottom=.23, left=.075, right=.97, wspace=.27)
    return [(fig, "combined_VR_selected_closure")]


def make_comparison_figures(result, config):
    """Return sixteen VR figures and five optional nominal MC SR figures."""
    global _opt_ORDER
    _opt_ORDER = tuple(config.get("candidate_order", result["specs"]))
    if not _opt_ORDER or len(set(_opt_ORDER)) != len(_opt_ORDER) or set(_opt_ORDER) != set(result["specs"]):
        raise ValueError("Candidate order must name every exclusive displacement specification exactly once.")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 10, "axes.titlesize": 11, "savefig.dpi": 170,
                         "figure.max_open_warning": 30}):
        figures = _opt_distribution_figures(result, config)
        figures += _opt_closure_figure(result, config)
        figures += _opt_scan_figures(result, config)
        figures += _opt_grid_figures(result, config)
        figures += _opt_composition_figures(result, config)
        figures += _opt_correlation_figure(result, config)
        figures += _candidate_purity_figure(result, config)
        figures += _candidate_combined_closure(result, config)
    mass_min, mass_max = float(config.get("mass_min", 150.)), config.get("mass_max")
    mass = (f"M(LJ,LJ) ≥ {mass_min:g} GeV" if mass_max is None else
            f"{mass_min:g} ≤ M(LJ,LJ) < {float(mass_max):g} GeV")
    baseline = f"{mass}; partner isolation < 0.5; exclusive displacement categories are stated above; no |Δφ| event precut."
    for figure, stem in figures:
        for caption in figure.texts:
            if caption.get_position()[1] < .1:
                caption.set_text(baseline + "\n" + caption.get_text())
        _opt_separate_y_ticks(figure)
    if "sr_reference" in result:
        figures += make_sr_reference_figures(result["sr_reference"], config)
    return figures


def make_sr_reference_figures(result, config):
    """Render the single nominal MC SR baseline, restoring VR plotting globals."""
    global _opt_ORDER, _opt_SAMPLES
    previous_order, previous_samples = _opt_ORDER, _opt_SAMPLES
    _opt_ORDER, _opt_SAMPLES = ("SR",), (("MC", "SR"),)
    try:
        if set(result["specs"]) != {"SR"}:
            raise ValueError("The nominal SR reference requires one specification named SR.")
        sr_config = dict(config, mass_min=150., mass_max=None)
        with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                             "axes.labelsize": 10, "axes.titlesize": 11, "savefig.dpi": 170,
                             "figure.max_open_warning": 30}):
            figures = _opt_distribution_figures(result, sr_config)
            figures += _opt_closure_figure(result, sr_config)
            figures += _opt_scan_figures(result, sr_config)
            figures += _opt_grid_figures(result, sr_config)
        baseline = "Nominal DD MC SR: M(LJ,LJ) ≥ 150 GeV; partner isolation < 0.5; no |Δφ| event precut."
        for figure, stem in figures:
            for caption in figure.texts:
                if caption.get_position()[1] < .1:
                    text = caption.get_text().replace("Hatching: DATA S/B criterion failed; ", "")
                    text = baseline + "\n" + text
                    caption.set_text("\n".join(textwrap.fill(line, width=105) for line in text.splitlines()))
                    caption.set_fontsize(8)
            _opt_separate_y_ticks(figure)
        return figures
    finally:
        _opt_ORDER, _opt_SAMPLES = previous_order, previous_samples


def _opt_data_approved(row):
    """Fail closed when the per-working-point DATA gate is absent or undefined."""
    approved = row.get("approved", False)
    return bool(pd.notna(approved) and bool(approved))


def _opt_mask_distribution(array, purity, xedges, yedges):
    """Hide DATA before normalization, failing closed on a mismatched MC mask."""
    for key, edges in [('xedges', xedges), ('yedges', yedges)]:
        if not np.array_equal(np.asarray(purity[key]), np.asarray(edges)):
            raise ValueError(f'DATA distribution purity mask has different {key}.')
    approved = np.asarray(purity['approved'])
    array = np.asarray(array, dtype=float)
    if approved.dtype != np.dtype(bool) or approved.shape != array.shape:
        raise ValueError('DATA distribution purity mask has an invalid shape or dtype.')
    return np.where(approved, array, np.nan)


# ========================================================================
# abcd_local_scan_plots: original study API
# ========================================================================
"""Readable, simultaneous WP comparisons for the local-scan notebooks.

This layer consumes validated tables and MC planes. It never opens DATA files,
reselects WPs using DATA, or transfers a VR correction to the nominal SR.
"""
import textwrap as finalize_local_scan_plots_textwrap

import matplotlib.pyplot as finalize_local_scan_plots_plt
from matplotlib.colors import LogNorm as finalize_local_scan_plots_LogNorm; from matplotlib.colors import Normalize as finalize_local_scan_plots_Normalize; from matplotlib.colors import TwoSlopeNorm as finalize_local_scan_plots_TwoSlopeNorm
from matplotlib.lines import Line2D as finalize_local_scan_plots_Line2D
from matplotlib.patches import Rectangle as finalize_local_scan_plots_Rectangle
from matplotlib.ticker import ScalarFormatter as finalize_local_scan_plots_ScalarFormatter
import numpy as finalize_local_scan_plots_np
import pandas as finalize_local_scan_plots_pd


finalize_local_scan_plots_SCOPES = ('SR', 'VR1_DD', 'VR1_PP', 'VR1_DP', 'VR1_PD', 'VR2_PP', 'VR2_DP', 'VR2_PD')
finalize_local_scan_plots_COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#666666')
finalize_local_scan_plots_COMPONENT_COLORS = {'QCD': '#E69F00', 'DY': '#0072B2', 'TTJets': '#009E73', 'Diboson': '#CC79A7'}


def finalize_local_scan_plots__frame(result, name, **filters):
    frame = result['tables'].get(name, finalize_local_scan_plots_pd.DataFrame()).copy()
    for key, value in filters.items():
        if key not in frame:
            return frame.iloc[:0]
        frame = frame.loc[frame[key].eq(value)]
    return frame


def finalize_local_scan_plots__scope_name(scope):
    return 'Nominal DD SR · MC' if scope == 'SR' else scope.replace('_', ' ')


def finalize_local_scan_plots__points(result):
    points = finalize_local_scan_plots__frame(result, 'working_points').sort_values('rank')
    selected = points.loc[points.is_top5 | points.is_reference].copy()
    return points, selected


def finalize_local_scan_plots__caption(fig, text):
    wrapped = '\n'.join(finalize_local_scan_plots_textwrap.fill(line, 176) for line in text.splitlines())
    fig.text(.5, .018, wrapped, ha='center', va='bottom', fontsize=9, color='#444444')


def finalize_local_scan_plots__canvas(title, rows=1, cols=1, bottom=.17, top=.85, wspace=.35, hspace=.55):
    fig, axes = finalize_local_scan_plots_plt.subplots(rows, cols, figsize=(16, 9), squeeze=False)
    fig.suptitle('2μ2e · ' + title, fontsize=20, x=.055, ha='left', y=.965)
    fig.subplots_adjust(left=.075, right=.955, top=top, bottom=bottom, wspace=wspace, hspace=hspace)
    for ax in axes.ravel():
        ax.tick_params(labelsize=9)
        ax.set_axisbelow(True)
    return fig, axes


def finalize_local_scan_plots__number(value, precision=2):
    if not finalize_local_scan_plots_np.isfinite(value):
        return '—'
    if value != 0 and (abs(value) < .01 or abs(value) >= 10000):
        return f'{value:.1e}'
    return f'{value:.{precision}f}'


def finalize_local_scan_plots__grid(frame, column, config):
    xs, ys = config['local_iso'], config['local_dphi']
    result = finalize_local_scan_plots_np.full((len(ys), len(xs)), finalize_local_scan_plots_np.nan)
    for row in frame.itertuples():
        ix = finalize_local_scan_plots_np.flatnonzero(finalize_local_scan_plots_np.isclose(xs, row.tx))
        iy = finalize_local_scan_plots_np.flatnonzero(finalize_local_scan_plots_np.isclose(ys, row.ty))
        if len(ix) == len(iy) == 1:
            result[iy[0], ix[0]] = getattr(row, column)
    return result


def finalize_local_scan_plots__draw_grid(ax, frame, column, config, *, title='', norm=None, cmap='viridis', errors=None, points=None):
    values = finalize_local_scan_plots__grid(frame, column, config)
    ax.set_facecolor('#e5e5e5')
    image = ax.imshow(finalize_local_scan_plots_np.ma.masked_invalid(values), origin='lower', aspect='auto', cmap=cmap, norm=norm)
    ax.set(xticks=range(4), xticklabels=[f'{x:.2f}' for x in config['local_iso']],
           yticks=range(4), yticklabels=[f'{y:.1f}' for y in config['local_dphi']], title=title)
    ax.set_xlabel('Mu isolation boundary', fontsize=9)
    ax.set_ylabel('|Δφ| boundary', fontsize=9)
    error_values = finalize_local_scan_plots__grid(frame, errors, config) if errors else None
    for iy in range(4):
        for ix in range(4):
            value = values[iy, ix]
            label = finalize_local_scan_plots__number(value)
            if error_values is not None and finalize_local_scan_plots_np.isfinite(value):
                label += '\n±' + finalize_local_scan_plots__number(error_values[iy, ix])
            ax.text(ix, iy, label, ha='center', va='center', fontsize=8.5,
                    color='#111111', bbox=dict(facecolor='white', alpha=.64, edgecolor='none', pad=.8))
    if points is not None:
        for point in points.itertuples():
            ix = list(config['local_iso']).index(point.tx)
            iy = list(config['local_dphi']).index(point.ty)
            if point.is_top5:
                ax.add_patch(finalize_local_scan_plots_Rectangle((ix-.47, iy-.47), .94, .94, fill=False, lw=2, edgecolor='black'))
            if point.is_reference:
                ax.plot(ix+.33, iy+.33, marker='*', color='white', mec='black', ms=10)
    return image


def finalize_local_scan_plots__map_page(result, config, name, title, column, source='MC', scopes=finalize_local_scan_plots_SCOPES,
              errors=None, norm=None, cmap='viridis', caption=''):
    fig, axes = finalize_local_scan_plots__canvas(title, 2, 4, bottom=.18, top=.84, hspace=.58)
    points, _ = finalize_local_scan_plots__points(result)
    image = None
    for ax, scope in zip(axes.ravel(), scopes):
        frame = finalize_local_scan_plots__frame(result, 'closure', scope=scope, source=source)
        image = finalize_local_scan_plots__draw_grid(ax, frame, column, config, title=finalize_local_scan_plots__scope_name(scope), norm=norm,
                           cmap=cmap, errors=errors, points=points)
    for ax in axes.ravel()[len(scopes):]:
        ax.axis('off')
    if image is not None:
        fig.colorbar(image, ax=list(axes.ravel()[:len(scopes)]), fraction=.018, pad=.025,
                     label=column.replace('_', ' '))
    finalize_local_scan_plots__caption(fig, caption + '\nBlack outlines: MC-ranked top five; star: reference. Gray cells: undefined or gated DATA. ')
    return fig, name


def finalize_local_scan_plots__overview(result, config):
    points, selected = finalize_local_scan_plots__points(result)
    ranking = finalize_local_scan_plots__frame(result, 'ranking').sort_values('rank')
    fig, axes = finalize_local_scan_plots__canvas('Local WP comparison: definitions and reading guide')
    ax = axes[0, 0]
    ax.axis('off')
    sections = [
        ('Fixed analysis selections', 'Nominal MC SR: DD, Mjj ≥ 150 GeV.  VR1: Mjj < 150 GeV, DD / PP / DP / PD.\nVR2: 150 ≤ Mjj < 300 GeV, PP / DP / PD.  All selections: 0 ≤ partner isolation < 0.5.'),
        ('16 native working points', 'Mu isolation = {0.20, 0.25, 0.30, 0.40}; |Δφ| = {2.4, 2.6, 2.7, 2.8}.\nA: low isolation / high |Δφ|; B: high / high; C: low / low; D: high / low.'),
        ('MC-only candidate ranking', 'Rank the median Z(WP)/Z(reference) over 60 separate matching-decay signal hypotheses.\nReference: (0.30, 2.6). Signal hypotheses are never added together; top five are fixed before VR inspection.'),
        ('Plot definitions', 'Event yield: A and ABCD prediction BC/D.  Closure: κ = A/(BC/D), with propagated uncertainty.\nContamination: maximum S/BMC over 120 hypotheses.  Effective count: Neff = (sumw)²/sumw².'),
        ('Interpretation', 'Nearby WPs share events. Toy frequencies are conditional moment-resampling diagnostics.\nVR representativeness requires comparison of all four ABCD compositions. Nominal DATA remains blind.')]
    for y, (heading, body) in zip(finalize_local_scan_plots_np.linspace(.95, .10, len(sections)), sections):
        ax.text(0, y, heading, weight='bold', fontsize=14, transform=ax.transAxes, va='top')
        ax.text(0, y-.055, body, fontsize=12, transform=ax.transAxes, va='top', linespacing=1.5)
    finalize_local_scan_plots__caption(fig, 'This notebook compares the fixed local domain; it does not establish a global optimum or adopt a final WP.')
    return fig, 'cover'


def finalize_local_scan_plots__ranking_table(result, config):
    ranking = finalize_local_scan_plots__frame(result, 'ranking').sort_values('rank')
    bg = finalize_local_scan_plots__frame(result, 'background_grid').set_index('wp')
    fig, axes = finalize_local_scan_plots__canvas('All 16 WPs: sensitivity and MC support', bottom=.17)
    ax = axes[0, 0]
    ax.axis('off')
    rows = []
    for row in ranking.itertuples():
        b = bg.loc[row.wp]
        rows.append([str(int(row.rank)), row.wp + (' ★' if row.is_reference else ''), f'{row.tx:.2f}', f'{row.ty:.1f}',
                     f'{row.z_known_b_ratio_median:.3f}', f'{row.z_known_b_ratio_q10:.2f}–{row.z_known_b_ratio_q90:.2f}',
                     f'{row.b:.2f} ± {row.sigma_b:.2f}', f'{row.neff_b:.2f}',
                     finalize_local_scan_plots__number(b.min_sideband_neff), f'{b.kappa:.3f} ± {b.kappa_error:.3f}'])
    headers = ['Rank', 'WP', 'Iso', '|Δφ|', 'Median\nZ ratio', '10–90%\nmodel spread', 'SR background A', 'Neff(A)', 'Min Neff\n(B,C,D)', 'MC SR κ']
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center',
                     colWidths=[.045, .06, .05, .05, .09, .12, .17, .08, .10, .17])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.83)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('#dddddd')
        if i == 0:
            cell.set_facecolor('#17324d'); cell.set_text_props(color='white', weight='bold')
            cell.set_height(cell.get_height() * 1.65)
        elif i <= 5:
            cell.set_facecolor('#e3f0fb')
        else:
            cell.set_facecolor('#f4f4f4' if i % 2 else 'white')
    finalize_local_scan_plots__caption(fig, 'Blue rows are the five MC-ranked display candidates. ★ marks the reference. Z uses nominal MC SR A yields.\n10–90% intervals describe differences among signal hypotheses, not statistical confidence intervals. Low effective MC counts remain explicit.')
    return fig, 'ranking_table'


def finalize_local_scan_plots__sr_maps(result, config):
    fig, axes = finalize_local_scan_plots__canvas('Nominal MC SR: gain and statistical support', 2, 2, bottom=.18)
    ranking = finalize_local_scan_plots__frame(result, 'ranking')
    points, _ = finalize_local_scan_plots__points(result)
    panels = [('z_known_b_ratio_median', 'Median known-background Z ratio', None),
              ('z_mc_stat_diagnostic_ratio_median', 'MC-statistical auxiliary Z ratio', None),
              ('b', 'Background MC A yield', finalize_local_scan_plots_LogNorm(vmin=ranking.b.min(), vmax=ranking.b.max())),
              ('neff_b', 'Background MC A effective count', None)]
    for ax, (column, title, norm) in zip(axes.ravel(), panels):
        image = finalize_local_scan_plots__draw_grid(ax, ranking, column, config, title=title, norm=norm, points=points)
        fig.colorbar(image, ax=ax, fraction=.035, pad=.03)
    finalize_local_scan_plots__caption(fig, 'Black outlines mark the top five known-background WPs; the star marks the reference.\nThe auxiliary Z incorporates background sumw² through an effective on/off diagnostic; neither metric is a fitted ABCD discovery significance.')
    return fig, 'sr_metric_maps'


def finalize_local_scan_plots__signal_pages(result, config):
    frame = finalize_local_scan_plots__frame(result, 'signal_significance', matching_decay=True)
    points, _ = finalize_local_scan_plots__points(result)
    ordered = points.sort_values('rank')
    matrix = frame.pivot(index='sample', columns='wp', values='z_known_b_ratio').reindex(columns=ordered.wp)
    figures = []
    for index, names in enumerate(finalize_local_scan_plots_np.array_split(matrix.index.to_numpy(), 2), 1):
        fig, axes = finalize_local_scan_plots__canvas(f'Individual signal hypotheses: Z ratio ({index}/2)', bottom=.18)
        ax = axes[0, 0]
        values = matrix.loc[names].to_numpy(float)
        image = ax.imshow(values, aspect='auto', vmin=0, vmax=max(4., float(finalize_local_scan_plots_np.nanmax(values))), cmap='viridis')
        labels = [name.replace('2Mu2E_', '').replace('GeV', '').replace('mm', '').replace('p', '.').replace('_', ' / ') for name in names]
        ax.set(xticks=range(len(ordered)), xticklabels=[f'{r.wp}\n{r.tx:.2f},{r.ty:.1f}' for r in ordered.itertuples()],
               yticks=range(len(names)), yticklabels=labels)
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='x', labelsize=8)
        for x in range(min(5, len(ordered))):
            ax.add_patch(finalize_local_scan_plots_Rectangle((x-.48, -.48), .96, len(names)-.04, fill=False, edgecolor='white', lw=1.5))
        fig.subplots_adjust(left=.20, right=.91)
        fig.colorbar(image, ax=ax, fraction=.025, pad=.02, label='Z(WP) / Z(reference)')
        finalize_local_scan_plots__caption(fig, 'Rows are separate matching-decay models, labeled parent mass [GeV] / dark-photon mass [GeV] / lifetime [mm].\nColumns are ordered by median gain; white outlines identify the top five. The hypotheses are not summed.')
        figures.append((fig, f'signal_gain_models_{index}'))
    return figures


def finalize_local_scan_plots__forest(result, config, source):
    _, selected = finalize_local_scan_plots__points(result)
    scopes = finalize_local_scan_plots_SCOPES if source == 'MC' else finalize_local_scan_plots_SCOPES[1:]
    fig, axes = finalize_local_scan_plots__canvas(f'Top five + reference: {source} closure', 2, 4, bottom=.20)
    labels = [f'{r.wp}  ({r.tx:.2f}, {r.ty:.1f})' for r in selected.itertuples()]
    for ax, scope in zip(axes.ravel(), scopes):
        frame = finalize_local_scan_plots__frame(result, 'closure', scope=scope, source=source).set_index('wp').reindex(selected.wp)
        y = finalize_local_scan_plots_np.arange(len(selected))
        ax.axvspan(.9, 1.1, color='#b9d8ba', alpha=.4)
        ax.axvline(1, color='#555555', ls='--', lw=1)
        for j, (_, row) in enumerate(frame.iterrows()):
            if finalize_local_scan_plots_np.isfinite(row.kappa):
                ax.errorbar(row.kappa, j, xerr=row.kappa_error if finalize_local_scan_plots_np.isfinite(row.kappa_error) else None,
                            fmt='o', color=finalize_local_scan_plots_COLORS[min(j, 5)], capsize=3)
            else:
                ax.text(.03, j, 'gated / undefined', transform=ax.get_yaxis_transform(), fontsize=8)
        ax.set(yticks=y, yticklabels=labels, xlabel='κ = A / (BC/D)', title=finalize_local_scan_plots__scope_name(scope))
        ax.invert_yaxis(); ax.grid(alpha=.2)
        if source == 'MC':
            ax.set_xscale('symlog', linthresh=.3)
    for ax in axes.ravel()[len(scopes):]: ax.axis('off')
    finalize_local_scan_plots__caption(fig, 'The green band is a ±10% visual reference, not an acceptance test. Error bars use the existing independent-region sumw² propagation.\nMC panels use a symmetric-log x axis; large or small MC κ values can reflect sparse sidebands. Missing DATA values stay gated.')
    return fig, 'top_kappa_' + source.lower()


def finalize_local_scan_plots__yield_page(result, config):
    _, selected = finalize_local_scan_plots__points(result)
    fig, axes = finalize_local_scan_plots__canvas('Top five + reference: A versus BC/D', 2, 4, bottom=.22)
    for ax, scope in zip(axes.ravel(), finalize_local_scan_plots_SCOPES):
        source = 'MC' if scope == 'SR' else 'DATA'
        frame = finalize_local_scan_plots__frame(result, 'closure', scope=scope, source=source).set_index('wp').reindex(selected.wp)
        x = finalize_local_scan_plots_np.arange(len(frame))
        ax.errorbar(x-.07, frame.A, yerr=frame.A_error, fmt='o', color='#222222', capsize=2, label='A')
        ax.errorbar(x+.07, frame.pred, yerr=frame.pred_error, fmt='s', color='#D55E00', capsize=2, label='BC/D')
        ax.set(xticks=x, xticklabels=[f'{r.wp}\n{r.tx:.2f},{r.ty:.1f}' for r in selected.itertuples()],
               ylabel='Events / weighted yield', title=finalize_local_scan_plots__scope_name(scope) + f' · {source}')
        ax.tick_params(axis='x', labelsize=8); ax.set_yscale('symlog', linthresh=1.)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=9)
    finalize_local_scan_plots__caption(fig, 'Nominal SR contains MC only; VR panels contain only DATA values passing the integrated 120-hypothesis S/B gate.\nThe symmetric-log y axis retains uncertainty bars crossing zero. Neighbouring WPs reuse events; the points are correlated.')
    return fig, 'top_yields'


def finalize_local_scan_plots__support_page(result, config, source, name, title, value_kind):
    scopes = finalize_local_scan_plots_SCOPES if source == 'MC' else finalize_local_scan_plots_SCOPES[1:]
    fig, axes = finalize_local_scan_plots__canvas(title, 2, 4, bottom=.22, top=.84)
    points, _ = finalize_local_scan_plots__points(result)
    points = points.sort_values('rank')
    columns = ['neff_'+r for r in 'ABCD'] if value_kind == 'neff' else list('ABCD')
    all_values = finalize_local_scan_plots__frame(result, 'closure', source=source)[columns].to_numpy(float)
    finite = all_values[finalize_local_scan_plots_np.isfinite(all_values)]
    norm = finalize_local_scan_plots_LogNorm(vmin=.5, vmax=max(10., float(finite.max())) if len(finite) else 10.)
    cmap = finalize_local_scan_plots_plt.get_cmap('viridis').copy()
    cmap.set_bad('#dedede')
    for ax, scope in zip(axes.ravel(), scopes):
        frame = finalize_local_scan_plots__frame(result, 'closure', scope=scope, source=source).set_index('wp').reindex(points.wp)
        values = frame[columns].to_numpy(float).T
        image = ax.imshow(finalize_local_scan_plots_np.ma.masked_invalid(values), origin='lower', aspect='auto', cmap=cmap, norm=norm)
        ax.set(yticks=range(4), yticklabels=list('ABCD'), xticks=range(len(points)), xticklabels=points.wp,
               title=finalize_local_scan_plots__scope_name(scope))
        ax.tick_params(axis='x', rotation=90, labelsize=7)
        for iy in range(4):
            for ix in range(len(points)):
                value = values[iy, ix]
                if not finalize_local_scan_plots_np.isfinite(value):
                    label = '×'
                elif value >= 1000:
                    label = f'{value/1000:.1f}k'
                elif value_kind == 'counts' or value >= 10:
                    label = f'{value:.0f}'
                else:
                    label = f'{value:.1f}'
                if finalize_local_scan_plots_np.isfinite(value) and value < 10:
                    label += '!'
                ax.text(ix, iy, label, ha='center', va='center', fontsize=5.9,
                        color='#111111', bbox=dict(facecolor='white', alpha=.65, edgecolor='none', pad=.15))
    for ax in axes.ravel()[len(scopes):]: ax.axis('off')
    fig.colorbar(image, ax=list(axes.ravel()[:len(scopes)]), fraction=.018, pad=.02,
                 label='MC effective count' if value_kind == 'neff' else 'Gated DATA event count')
    finalize_local_scan_plots__caption(fig, 'Columns follow the MC sensitivity ranking; colors use one common scale. ! flags counts below 10; k means ×1000.\n× marks blocked/undefined entries; an explicit 0 remains a valid zero count. MC values are effective counts, not generated event counts.')
    return fig, name


def finalize_local_scan_plots__contamination(result, config):
    fig, axes = finalize_local_scan_plots__canvas('All 16 WPs: maximum signal contamination', 2, 4, bottom=.19)
    points, _ = finalize_local_scan_plots__points(result)
    for ax, scope in zip(axes.ravel(), finalize_local_scan_plots_SCOPES):
        frame = finalize_local_scan_plots__frame(result, 'closure', scope=scope, source='MC')
        image = finalize_local_scan_plots__draw_grid(ax, frame, 'max_S_B', config, title=finalize_local_scan_plots__scope_name(scope),
                           norm=finalize_local_scan_plots_LogNorm(vmin=1e-4, vmax=100), points=points)
    fig.colorbar(image, ax=list(axes.ravel()), fraction=.018, pad=.02, label='max S/BMC over 120 models and ABCD')
    finalize_local_scan_plots__caption(fig, 'Contamination is an MC diagnostic, including the SR. The DATA gate requires S/BMC < 0.10 separately in every ABCD region for all 120 models.\nPassing this gate does not establish closure. Colors span 10⁻⁴–10²; numerical labels retain values outside that range.')
    return fig, 'contamination_maps'


def finalize_local_scan_plots__contamination_regions(result, config):
    _, selected = finalize_local_scan_plots__points(result)
    fig, axes = finalize_local_scan_plots__canvas('Top five + reference: contamination in each ABCD region', 2, 4, bottom=.22)
    for ax, scope in zip(axes.ravel(), finalize_local_scan_plots_SCOPES):
        frame = finalize_local_scan_plots__frame(result, 'purity', scope=scope)
        for j, region in enumerate('ABCD'):
            subset = frame.loc[frame.region.eq(region)].set_index('wp').reindex(selected.wp)
            ax.plot(range(len(selected)), subset.max_S_B, marker='o', color=finalize_local_scan_plots_COLORS[j], label=region)
        ax.axhline(.1, color='black', ls='--', lw=1)
        ax.set(yscale='log', xticks=range(len(selected)), xticklabels=selected.wp,
               ylabel='Maximum S/BMC', title=finalize_local_scan_plots__scope_name(scope))
        ax.grid(alpha=.2)
    axes[0, 0].legend(ncol=4, fontsize=8)
    finalize_local_scan_plots__caption(fig, 'Every marker is the maximum over 120 distinct signal hypotheses in that region. The maximizing model can change with WP.\nThe dashed 10% line is the integrated DATA-display criterion, not a fitted uncertainty or a cut on MC SR visibility.')
    return fig, 'contamination_regions'




def finalize_local_scan_plots__composition_page(result, config, scopes, name):
    _, selected = finalize_local_scan_plots__points(result)
    fig, axes = finalize_local_scan_plots__canvas('Top five + reference: ABCD background composition', 2, 2, bottom=.23)
    components = ('QCD','DY','TTJets','Diboson')
    for ax, scope in zip(axes.ravel(), scopes):
        frame = finalize_local_scan_plots__frame(result, 'composition', scope=scope)
        positions = finalize_local_scan_plots_np.arange(len(selected)*4)
        bottoms = finalize_local_scan_plots_np.zeros(len(positions))
        for component in components:
            subset = frame.loc[frame.component.eq(component)].set_index(['wp','region'])
            heights = finalize_local_scan_plots_np.array([subset.loc[(wp,r),'fraction'] if (wp,r) in subset.index else 0 for wp in selected.wp for r in 'ABCD'],float)
            ax.bar(positions, heights, bottom=bottoms, color=finalize_local_scan_plots_COMPONENT_COLORS[component], label=component, width=.88)
            bottoms += heights
        for x in finalize_local_scan_plots_np.arange(3.5,len(positions),4): ax.axvline(x,color='white',lw=2)
        ax.set(xticks=positions, xticklabels=list('ABCD')*len(selected), ylim=(0,1.03),
               ylabel='MC central fraction', title=finalize_local_scan_plots__scope_name(scope))
        ax.tick_params(axis='x',labelsize=7)
        for j, row in enumerate(selected.itertuples()):
            ax.text(4*j+1.5,-.15,row.wp,ha='center',transform=ax.get_xaxis_transform(),fontsize=9)
    for ax in axes.ravel()[len(scopes):]: ax.axis('off')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.085),ncol=4,fontsize=10)
    finalize_local_scan_plots__caption(fig, 'Composition must be compared in all four ABCD regions at the same WP. These are low-statistics MC central-value fractions.\nMatching a dominant process in A alone does not establish VR-to-SR representativeness; no correction factor is transferred here.')
    return fig,name


def finalize_local_scan_plots__transfer_page(result, config):
    points,_=finalize_local_scan_plots__points(result)
    points=points.sort_values('rank')
    frame=finalize_local_scan_plots__frame(result,'transfer_summary')
    fig,axes=finalize_local_scan_plots__canvas('VR-to-SR applicability: composition distance in ABCD',2,4,bottom=.22)
    for ax,scope in zip(axes.ravel(),finalize_local_scan_plots_SCOPES[1:]):
        subset=frame.loc[frame.scope.eq(scope)]
        matrix=subset.pivot(index='region',columns='wp',values='total_variation').reindex(index=list('ABCD'),columns=points.wp)
        im=ax.imshow(finalize_local_scan_plots_np.ma.masked_invalid(matrix.to_numpy(float)),origin='lower',aspect='auto',vmin=0,vmax=1,cmap='magma_r')
        ax.set(xticks=range(len(points)),xticklabels=points.wp,yticks=range(4),yticklabels=list('ABCD'),title=finalize_local_scan_plots__scope_name(scope))
        ax.tick_params(axis='x',rotation=90,labelsize=7)
    axes.ravel()[-1].axis('off')
    fig.colorbar(im,ax=list(axes.ravel()[:-1]),fraction=.018,pad=.02,label='Total variation: ½ Σprocess |fVR − fSR|')
    finalize_local_scan_plots__caption(fig, 'A value of 0 means identical MC central composition and 1 means disjoint process composition. Columns follow MC sensitivity rank.\nThis distance is descriptive, has no calibrated uncertainty here, and is not a transfer factor or a selection requirement. Compare it with effective-count support.')
    return fig,'transfer_composition'


def finalize_local_scan_plots__dominance_page(result, config):
    _,selected=finalize_local_scan_plots__points(result)
    frame=finalize_local_scan_plots__frame(result,'dominance_summary')
    fig,axes=finalize_local_scan_plots__canvas('Large-weight sample dominance: nominal MC SR',2,2,bottom=.24)
    for ax,region in zip(axes.ravel(),'ABCD'):
        subset=frame.loc[frame.scope.eq('SR') & frame.region.eq(region)].set_index('wp').reindex(selected.wp)
        x=finalize_local_scan_plots_np.arange(len(selected))
        ax.bar(x,subset.largest_sample_fraction,color=[finalize_local_scan_plots_COLORS[min(j,5)] for j in x],width=.65)
        ax.set(xticks=x,xticklabels=selected.wp,ylim=(0,1.17),ylabel='Largest single-sample fraction',title='ABCD region '+region)
        for j,row in enumerate(subset.itertuples()):
            name=str(row.largest_sample).replace('QCD_','QCD ').replace('DYJetsToLL_','DY ').replace('Pt','pT ')
            ax.text(j,min(1.06,float(row.largest_sample_fraction)+.025),finalize_local_scan_plots_textwrap.fill(name,18),ha='center',va='bottom',fontsize=7)
        ax.grid(axis='y',alpha=.2)
    finalize_local_scan_plots__caption(fig, 'Dominance identifies which production sample controls a yield; it is not a statistical uncertainty estimate.\nThe underlying sample may contain only a few effective events. Removing a physical background sample is not a valid bootstrap procedure.')
    return fig,'sample_dominance'










def finalize_local_scan_plots__decision_page(result,config):
    ranking=finalize_local_scan_plots__frame(result,'ranking').sort_values('rank')
    best=ranking.iloc[0]
    fig,axes=finalize_local_scan_plots__canvas('What the local comparison can establish')
    ax=axes[0,0]; ax.axis('off')
    sections=[('Current MC preference',f'{best.wp}: mu isolation={best.tx:.2f}, |Δφ|={best.ty:.1f}; median known-background Z ratio={best.z_known_b_ratio_median:.3f}.\nRead the full signal-model distribution and common-fluctuation diagnostics before interpreting this preference.'),
              ('Background-estimation support','Inspect A and B/C/D independently: yields, effective counts, process fractions and dominant samples.\nAn apparently favorable Z can depend on a few large-weight MC contributions.'),
              ('VR evidence','Use the gated VR1 and VR2 κ patterns and neighbouring boundaries to identify what each VR can test.\nComposition similarity is descriptive; it does not establish a numerical κ transfer or a calibrated uncertainty.'),
              ('Next decision','A final WP requires a defensible ABCD background model and its uncertainty treatment.\nKeep statistically indistinguishable candidates together; retain the reference as a reproducible baseline.')]
    for y,(heading,body) in zip((.94,.70,.46,.22),sections):
        ax.text(.01,y,heading,fontsize=15,weight='bold',transform=ax.transAxes,va='top')
        ax.text(.01,y-.065,body,fontsize=12.5,transform=ax.transAxes,va='top',linespacing=1.6)
    finalize_local_scan_plots__caption(fig,'No nominal DATA is accessed. Top-five labels are determined by SR MC only; no final analysis WP is declared by this notebook.')
    return fig,'decision_summary'


def finalize_local_scan_plots_local_scan_figure_definitions():
    """Return the fixed notebook figure order and the purpose of each view."""
    entries=[('cover','Definitions and reading guide','Selections, WP domain, metric definitions and interpretation.'),
             ('ranking_table','All-WP ranking and MC support','Compare all 16 candidate boundaries in one numerical table.'),
             ('sr_metric_maps','SR sensitivity and support maps','View sensitivity, background yield and effective counts on common WP coordinates.'),
             ('signal_gain_models_1','Signal-model gains, part 1','Compare separate signal hypotheses across the ranked WPs.'),
             ('signal_gain_models_2','Signal-model gains, part 2','Complete the 60-hypothesis comparison without merging signal models.'),
             ('vr_data_kappa_maps','VR DATA closure maps','Compare seven gated VR κ landscapes and their propagated errors.'),
             ('mc_kappa_maps','MC closure maps','Compare nominal SR and all VR MC closure landscapes.'),
             ('top_kappa_data','Selected-WP DATA closure','Display the top five and reference with κ error bars.'),
             ('top_kappa_mc','Selected-WP MC closure','Expose sparse-sideband MC closure behaviour at the same candidate boundaries.'),
             ('top_yields','Observed A and ABCD prediction','Compare event counts and BC/D estimates simultaneously across candidates.'),
             ('mc_neff_maps','ABCD MC effective-count support','Identify unsupported quadrants separately from the A yield.'),
             ('data_counts_maps','ABCD DATA-count support','Show gated DATA counts and low-count flags for each quadrant.'),
             ('contamination_maps','All-WP contamination','Display the maximum S/BMC across all 120 hypotheses and ABCD regions.'),
             ('contamination_regions','Quadrant-resolved contamination','See which ABCD quadrant limits the DATA gate.'),
             ('mc_distributions','Conditional MC distributions','Overlay raw and normalized shapes with exact WP labels.'),
             ('composition_sr_vr1','SR and VR1 process fractions','Compare all four ABCD process compositions.'),
             ('composition_vr2','Remaining VR1 and VR2 process fractions','Complete the composition comparison with VR1 PD and all three VR2 categories.'),
             ('transfer_composition','VR-to-SR composition distance','Quantify descriptive process-fraction differences in each quadrant.'),
             ('sample_dominance','Large-weight sample dominance','Identify the production samples driving nominal SR yields.')]
    entries.extend(('scan_'+scope,finalize_local_scan_plots__scope_name(scope)+' 1D scans','Shared-axis event-yield, κ and contamination scans for selected WP families.') for scope in finalize_local_scan_plots_SCOPES)
    entries.extend([('toy_rankings','Ranking-stability diagnostics','Evaluate conditional rank frequencies under common MC moment fluctuations.'),
                    ('toy_covariance','Shared-event covariance','Show cross-WP correlations and unsupported-replica rates.'),
                    ('likelihood_scenarios','ABCD likelihood inputs','Define the expected likelihood and inspect actual input support; declared scenarios only.'),
                    ('decision_summary','Interpretation and next decision','State the current MC preference and the evidence needed for a final WP.')])
    return finalize_local_scan_plots_pd.DataFrame(entries,columns=['name','title','purpose'])


def finalize_local_scan_plots_make_local_scan_figures(result,config):
    """Build all notebook figures without loading samples or exposing DATA bins."""
    figures=[]
    with finalize_local_scan_plots_plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
                         'axes.labelsize':10,'pdf.fonttype':42,'figure.max_open_warning':0}):
        figures.extend([finalize_local_scan_plots__overview(result,config),finalize_local_scan_plots__ranking_table(result,config),finalize_local_scan_plots__sr_maps(result,config)])
        figures.extend(finalize_local_scan_plots__signal_pages(result,config))
        figures.append(finalize_local_scan_plots__map_page(result,config,'vr_data_kappa_maps','VR DATA: local κ landscapes ± propagated error','kappa',source='DATA',scopes=finalize_local_scan_plots_SCOPES[1:],errors='kappa_error',norm=finalize_local_scan_plots_TwoSlopeNorm(vmin=0,vcenter=1,vmax=2),cmap='coolwarm',caption='DATA colors saturate outside [0,2]; numerical values and errors are retained. The green-band target is only a visual reference.'))
        figures.append(finalize_local_scan_plots__map_page(result,config,'mc_kappa_maps','MC: local κ landscapes and sparse-sideband effects','kappa',source='MC',errors='kappa_error',norm=finalize_local_scan_plots_LogNorm(vmin=.01,vmax=100),cmap='coolwarm',caption='MC colors use a logarithmic range 0.01–100; labels retain full central values and propagated errors.'))
        figures.extend([finalize_local_scan_plots__forest(result,config,'DATA'),finalize_local_scan_plots__forest(result,config,'MC'),finalize_local_scan_plots__yield_page(result,config),
                        finalize_local_scan_plots__support_page(result,config,'MC','mc_neff_maps','ABCD MC statistical support: all 16 WPs','neff'),
                        finalize_local_scan_plots__support_page(result,config,'DATA','data_counts_maps','Gated ABCD DATA counts: all 16 WPs','counts'),
                        finalize_local_scan_plots__contamination(result,config),finalize_local_scan_plots__contamination_regions(result,config),(finalize_local_scan_plots__local_distributions_figure(result,config), 'mc_distributions'),
                        finalize_local_scan_plots__composition_page(result,config,('SR','VR1_DD','VR1_PP','VR1_DP'),'composition_sr_vr1'),
                        finalize_local_scan_plots__composition_page(result,config,('VR1_PD','VR2_PP','VR2_DP','VR2_PD'),'composition_vr2'),
                        finalize_local_scan_plots__transfer_page(result,config),finalize_local_scan_plots__dominance_page(result,config)])
        figures.extend((finalize_local_scan_plots__local_scan_figure(result,config,scope), 'scan_'+scope) for scope in finalize_local_scan_plots_SCOPES)
        figures.extend([(finalize_local_scan_plots__local_toy_rankings(result,config), 'toy_rankings'),(finalize_local_scan_plots__local_toy_covariance(result,config), 'toy_covariance'),(finalize_local_scan_plots__local_likelihood_figure(result,config), 'likelihood_scenarios'),finalize_local_scan_plots__decision_page(result,config)])
    expected=finalize_local_scan_plots_local_scan_figure_definitions().name.tolist()
    names=[name for _,name in figures]
    if names!=expected:
        raise AssertionError('The generated figures do not match the fixed notebook registry.')
    for fig,_ in figures:
        fig.canvas.draw()
    return figures


# Additional notebook views: abcd07_scan_fragment
import numpy as finalize_local_scan_plots_np
import pandas as finalize_local_scan_plots_pd
import matplotlib.pyplot as finalize_local_scan_plots_plt
from matplotlib.lines import Line2D as finalize_local_scan_plots_Line2D
finalize_local_scan_plots_common = _FINALIZE_NAMESPACES['abcd_finalize_common']


def finalize_local_scan_plots__scan_selected(result):
    points = result['tables']['working_points']
    selected = points.loc[points.is_top5 | points.is_reference].sort_values('rank').copy()
    selected['plot_color'] = [('#666666' if row.is_reference else finalize_local_scan_plots_COLORS[(int(row.rank) - 1) % 5])
                              for row in selected.itertuples()]
    markers = ['o', 's', '^', 'D', 'P']
    selected['plot_marker'] = [('X' if row.is_reference else markers[(int(row.rank) - 1) % 5])
                               for row in selected.itertuples()]
    return selected


def finalize_local_scan_plots__scan_wp_label(point):
    name = 'Ref' if point.is_reference else f'{point.wp} #{int(point.rank)}'
    return f'{name}: iso={point.tx:.2f}, |Δφ|={point.ty:.1f}'


def finalize_local_scan_plots__scan_point(frame, tx, ty):
    chosen = frame.loc[finalize_local_scan_plots_np.isclose(frame.tx, tx, rtol=0, atol=1e-12)
                       & finalize_local_scan_plots_np.isclose(frame.ty, ty, rtol=0, atol=1e-12)]
    if len(chosen) != 1:
        raise ValueError(f'Expected one scan point at ({tx}, {ty}).')
    return chosen.iloc[0]


def finalize_local_scan_plots__scan_scope_text(scope):
    if scope == 'SR':
        return 'Nominal DD; Mjj ≥ 150 GeV; MC only'
    vr, states = scope.split('_')
    if vr == 'VR2':
        return f'{states}; 150 ≤ MJJ < 300 GeV; gated DATA'
    mass = '0 ≤ Mjj < 150 GeV' if vr == 'VR1' else '150 ≤ Mjj < 300 GeV'
    return f'{vr} {states}; {mass}; gated DATA'


def finalize_local_scan_plots__scan_series(axis, x, values, errors, color, marker, linestyle, alpha=.88):
    y, uncertainty = finalize_local_scan_plots_np.asarray(values, float), finalize_local_scan_plots_np.asarray(errors, float)
    valid = finalize_local_scan_plots_np.isfinite(y) & finalize_local_scan_plots_np.isfinite(uncertainty) & (uncertainty >= 0)
    if valid.any():
        axis.errorbar(finalize_local_scan_plots_np.asarray(x), finalize_local_scan_plots_np.where(valid, y, finalize_local_scan_plots_np.nan),
                      yerr=finalize_local_scan_plots_np.where(valid, uncertainty, finalize_local_scan_plots_np.nan),
                      color=color, marker=marker, markersize=3.6, linestyle=linestyle,
                      linewidth=1.4, elinewidth=.9, capsize=2, alpha=alpha,
                      markerfacecolor=color if marker == 'o' else 'white')


def finalize_local_scan_plots__local_scan_figure(result, config, scope):
    """Native-threshold stability scans with shared-family curves and gate gaps."""
    selected = finalize_local_scan_plots__scan_selected(result)
    source = 'MC' if scope == 'SR' else 'DATA'
    full = result['tables']['full_closure']
    values = full.loc[full.scope.eq(scope) & full.source.eq(source)]
    purity = full.loc[full.scope.eq(scope) & full.source.eq('MC')]
    fig, axes = finalize_local_scan_plots_plt.subplots(3, 2, figsize=(16, 9), sharex='col',
                             gridspec_kw={'height_ratios': [1.3, 1., .9]})
    expanded = 'scan_iso_limits' in config or 'scan_dphi_limits' in config
    title = '1D stability scans' if expanded else 'local 1D comparisons'
    topology_label = '4μ' if config['topology'] == '4mu' else '2μ2e'
    scope_label = ('Intermediate MJJ VR · ' + scope.split('_')[1]
                   if scope.startswith('VR2_') else scope.replace('_', ' '))
    fig.suptitle(f'{topology_label} · {scope_label} · {title}', y=.987,
                 fontsize=18, fontweight='bold')
    fig.text(.5, .947, finalize_local_scan_plots__scan_scope_text(scope) + '; 0 ≤ partner isolation < 0.5',
             ha='center', fontsize=11)
    low_stat_seen, blocked_seen = False, False
    for column, direction in enumerate(('isolation', 'angle')):
        scan_field, fixed_field = ('tx', 'ty') if direction == 'isolation' else ('ty', 'tx')
        local = config['local_iso' if direction == 'isolation' else 'local_dphi']
        limit_key = 'scan_iso_limits' if direction == 'isolation' else 'scan_dphi_limits'
        limits = tuple(float(value) for value in config.get(limit_key, (local[0], local[-1])))
        if len(limits) != 2 or limits[0] >= limits[1]:
            raise ValueError('Each scan display range must contain increasing lower and upper limits.')
        available = finalize_local_scan_plots_np.sort(values[scan_field].unique().astype(float))
        thresholds = available[(available >= limits[0] - 1e-12) & (available <= limits[1] + 1e-12)]
        if len(thresholds) < 2:
            raise ValueError('A stability scan requires at least two stored native thresholds.')
        fixed_values = selected[fixed_field].drop_duplicates().tolist()
        family_handles = []
        all_counts, all_kappa, all_purity = [], [], []
        for family_index, fixed in enumerate(fixed_values):
            members = selected.loc[finalize_local_scan_plots_np.isclose(selected[fixed_field], fixed, rtol=0, atol=1e-12)]
            representative = next(members.itertuples())
            color = representative.plot_color
            rows, mc_rows = [], []
            for threshold in thresholds:
                tx, ty = (threshold, fixed) if direction == 'isolation' else (fixed, threshold)
                rows.append(finalize_local_scan_plots__scan_point(values, tx, ty))
                mc_rows.append(finalize_local_scan_plots__scan_point(purity, tx, ty))
            rows, mc_rows = finalize_local_scan_plots_pd.DataFrame(rows), finalize_local_scan_plots_pd.DataFrame(mc_rows)
            gate = finalize_local_scan_plots_np.ones(len(rows), bool) if source == 'MC' else rows.approved.to_numpy(bool)
            a = rows.A.to_numpy(float)
            p = rows.pred.to_numpy(float)
            kappa = rows.kappa.to_numpy(float)
            # Reassert gate masking before displaying any observed DATA quantity.
            for name in ('A', 'pred', 'kappa', 'A_error', 'pred_error', 'kappa_error'):
                if source == 'DATA' and not rows.loc[~rows.approved, name].isna().all():
                    raise AssertionError('Blocked DATA must remain hidden in scan figures.')
            finalize_local_scan_plots__scan_series(axes[0, column], thresholds, a, rows.A_error, color, 'o', '-')
            finalize_local_scan_plots__scan_series(axes[0, column], thresholds, p, rows.pred_error, color, 's', '--', .72)
            finalize_local_scan_plots__scan_series(axes[1, column], thresholds, kappa, rows.kappa_error, color, 'o', '-')
            ratio = mc_rows.max_S_B.to_numpy(float)
            positive = finalize_local_scan_plots_np.isfinite(ratio) & (ratio > 0)
            axes[2, column].plot(thresholds, finalize_local_scan_plots_np.where(positive, ratio, finalize_local_scan_plots_np.nan), 'o-', color=color,
                                 markersize=3.3, linewidth=1.4)
            valid_low_stat = gate & rows.low_stat.to_numpy(bool) & finalize_local_scan_plots_np.isfinite(kappa)
            if valid_low_stat.any():
                axes[1, column].scatter(thresholds[valid_low_stat], kappa[valid_low_stat],
                                       marker='x', color='#B52B25', s=55, linewidth=1.3, zorder=7)
                low_stat_seen = True
            blocked = ~gate
            if blocked.any():
                axes[1, column].plot(thresholds[blocked], finalize_local_scan_plots_np.full(blocked.sum(), .035 + .035 * family_index),
                                     'x', color='#666666', markersize=6,
                                     transform=axes[1, column].get_xaxis_transform(), clip_on=False)
                blocked_seen = True
            family_name = f'|Δφ|={fixed:.1f}' if direction == 'isolation' else f'iso={fixed:.2f}'
            family_handles.append(finalize_local_scan_plots_Line2D([0], [0], color=color, linewidth=2, label=family_name))
            all_counts.extend(finalize_local_scan_plots_np.r_[a[finalize_local_scan_plots_np.isfinite(a)], p[finalize_local_scan_plots_np.isfinite(p)]])
            all_kappa.extend(kappa[finalize_local_scan_plots_np.isfinite(kappa)])
            all_purity.extend(ratio[positive])
        axes[0, column].set_title('Isolation threshold scan' if direction == 'isolation' else '|Δφ| threshold scan',
                                  fontsize=12, pad=8)
        family_title = '|Δφ| held fixed' if direction == 'isolation' else 'Isolation held fixed'
        fig.legend(handles=family_handles, loc='upper left',
                   bbox_to_anchor=(.075 if column == 0 else .565, .924), ncol=2,
                   title=family_title, fontsize=9.1, title_fontsize=9.5,
                   frameon=False, borderpad=.25, columnspacing=1.6)
        if source == 'MC':
            axes[0, column].set_yscale('symlog', linthresh=1.)
        axes[0, column].set_ylim(bottom=0)
        axes[1, column].axhspan(.9, 1.1, color='#79A968', alpha=.18, zorder=0)
        axes[1, column].axhline(1., color='#5A7D4E', linewidth=.9, zorder=0)
        axes[2, column].axhline(.1, color='#B52B25', linestyle='--', linewidth=1.1)
        axes[2, column].set_yscale('log')
        if all_purity:
            axes[2, column].set_ylim(min(min(all_purity) * .6, .02), max(max(all_purity) * 1.7, .17))
        for point in selected.itertuples():
            row = finalize_local_scan_plots__scan_point(values, point.tx, point.ty)
            x = point.tx if direction == 'isolation' else point.ty
            allowed = source == 'MC' or bool(row.approved)
            for panel, field in ((0, 'A'), (1, 'kappa')):
                value = float(row[field])
                if allowed and finalize_local_scan_plots_np.isfinite(value):
                    axes[panel, column].scatter([x], [value], s=72, marker=point.plot_marker,
                                                color=point.plot_color, edgecolors='white', linewidth=.8, zorder=10)
            contamination = finalize_local_scan_plots__scan_point(purity, point.tx, point.ty).max_S_B
            if finalize_local_scan_plots_np.isfinite(contamination) and contamination > 0:
                axes[2, column].scatter([x], [contamination], s=65, marker=point.plot_marker,
                                        color=point.plot_color, edgecolors='white', linewidth=.8, zorder=10)
        isolation_label = 'Leading mu-LJ isolation threshold' if config['topology'] == '4mu' else 'Mu-LJ isolation threshold'
        label = isolation_label if direction == 'isolation' else '|Δφ| threshold [rad]'
        axes[2, column].set_xlabel(label, fontsize=11)
        ticks = thresholds
        if limit_key in config:
            proposed = ([0., .1, .2, .3, .4, .5, .75, 1.] if direction == 'isolation'
                        else [1., 1.5, 2., 2.4, 2.6, 2.8, 3.])
            ticks = [tick for tick in proposed if limits[0] - 1e-12 <= tick <= limits[1] + 1e-12]
        for axis in axes[:, column]:
            axis.set_xlim(*limits)
            axis.set_xticks(ticks)
            axis.grid(alpha=.2, linewidth=.7)
            axis.tick_params(labelsize=10)
    axes[0, 0].set_ylabel(('Events (symlog)' if source == 'MC' else 'Events') + '\nA / BC/D', fontsize=10)
    axes[1, 0].set_ylabel('κ = A / (BC/D)', fontsize=10)
    axes[2, 0].set_ylabel('Maximum MC S/B\n120 hypotheses × ABCD', fontsize=10)
    styles = [finalize_local_scan_plots_Line2D([0], [0], color='#222222', marker='o', label='Observed/direct A'),
              finalize_local_scan_plots_Line2D([0], [0], color='#222222', marker='s', markerfacecolor='white',
                     linestyle='--', label='ABCD prediction BC/D')]
    fig.legend(handles=styles, loc='upper center', bbox_to_anchor=(.5, .172), ncol=2,
               fontsize=9.5, frameon=False)
    handles = [finalize_local_scan_plots_Line2D([0], [0], color=point.plot_color, marker=point.plot_marker,
                      linestyle='None', markersize=7, label=finalize_local_scan_plots__scan_wp_label(point))
               for point in selected.itertuples()]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .075),
               ncol=3, fontsize=9.1, frameon=False, columnspacing=1.7, handletextpad=.5)
    flag_text = ('Red ×: fewer than 10 effective events/counts in an ABCD region. ' if low_stat_seen else '')
    flag_text += ('Gray × at panel base: DATA contamination gate blocked. ' if blocked_seen else '')
    fig.text(.5, .015, 'Curves use stored native thresholds in the shown ranges; zero is an axis limit, not an evaluated split. '
             'Joining lines are visual guides. Shared fixed-axis curves are drawn once; markers identify selected WPs.\n'
             + flag_text + 'κ band: ±10% reference. Bottom row is MC contamination even for DATA panels; red line is S/B=0.10.',
             ha='center', va='bottom', fontsize=8.7)
    fig.subplots_adjust(left=.075, right=.985, top=.80, bottom=.255, hspace=.18, wspace=.19)
    return fig


def finalize_local_scan_plots__scan_conditional_distribution(plane, tx, ty, axis):
    """Return native-bin MC moments with explicitly folded numeric flows."""
    if axis == 'isolation':
        boundary = finalize_local_scan_plots_common._comparison_split(plane['yedges'], ty)
        moments = [plane[key][1:, boundary:].sum(axis=1) for key in ('values', 'variances')]
        values = [moment[:-1].copy() for moment in moments]
        for value, moment in zip(values, moments):
            value[-1] += moment[-1]
        return plane['xedges'], values[0], values[1]
    boundary = finalize_local_scan_plots_common._comparison_split(plane['xedges'], tx)
    moments = [plane[key][1:boundary, :].sum(axis=0) for key in ('values', 'variances')]
    values = [moment[1:-1].copy() for moment in moments]
    for value, moment in zip(values, moments):
        value[0] += moment[0]
        value[-1] += moment[-1]
    return plane['yedges'], values[0], values[1]


def finalize_local_scan_plots__local_distributions_figure(result, config):
    """Conditional nominal-SR MC shapes and yields for the same highlighted WPs."""
    selected = finalize_local_scan_plots__scan_selected(result)
    plane = result['planes']['SR']
    fig, axes = finalize_local_scan_plots_plt.subplots(2, 2, figsize=(16, 9), sharex='col')
    fig.suptitle('2μ2e · nominal SR MC · conditional distributions', y=.987,
                 fontsize=18, fontweight='bold')
    fig.text(.5, .947, 'DD; Mjj ≥ 150 GeV; 0 ≤ partner isolation < 0.5; background MC only',
             ha='center', fontsize=11)
    for column, direction in enumerate(('isolation', 'angle')):
        field = 'ty' if direction == 'isolation' else 'tx'
        groups = selected[field].drop_duplicates().tolist()
        limits = (0., .5) if direction == 'isolation' else (2.2, finalize_local_scan_plots_np.pi)
        family_handles = []
        displayed_max, normalized_max = [], []
        for fixed in groups:
            members = selected.loc[finalize_local_scan_plots_np.isclose(selected[field], fixed, rtol=0, atol=1e-12)]
            point = next(members.itertuples())
            edges, values, variances = finalize_local_scan_plots__scan_conditional_distribution(plane, point.tx, point.ty, direction)
            centers = (edges[:-1] + edges[1:]) / 2
            visible = (centers >= limits[0]) & (centers <= limits[1])
            errors = finalize_local_scan_plots_np.sqrt(variances)
            color = point.plot_color
            axes[0, column].stairs(values, edges, color=color, linewidth=1.8)
            axes[0, column].errorbar(centers[visible], values[visible], yerr=errors[visible],
                                    color=color, linestyle='none', marker='o', markersize=3.2,
                                    elinewidth=1., capsize=2, alpha=.8)
            total = float(values.sum())
            if total > 0 and finalize_local_scan_plots_np.isfinite(total):
                normalized = values / total
                axes[1, column].stairs(normalized, edges, color=color, linewidth=1.8)
                normalized_max.extend(normalized[visible])
            displayed_max.extend((values + errors)[visible])
            identifiers = ', '.join('Ref' if row.is_reference else row.wp for row in members.itertuples())
            condition = f'|Δφ| ≥ {fixed:.1f}' if direction == 'isolation' else f'iso < {fixed:.2f}'
            family_handles.append(finalize_local_scan_plots_Line2D([0], [0], color=color, linewidth=2,
                                         label=f'{condition} [{identifiers}]'))
        axes[0, column].set_title('Isolation given the WP angle selection' if direction == 'isolation'
                                  else '|Δφ| given the WP isolation selection', fontsize=12, pad=10)
        axes[0, column].legend(handles=family_handles, loc='upper right', fontsize=9,
                               title='Shared conditional curves', title_fontsize=10, framealpha=.9)
        axes[0, column].set_yscale('symlog', linthresh=1.)
        axes[0, column].set_ylim(0., max(displayed_max) * 1.7 if displayed_max else 1.)
        axes[1, column].set_ylim(0., max(normalized_max) * 1.18 if normalized_max else 1.)
        cutoff = 'tx' if direction == 'isolation' else 'ty'
        for threshold in selected[cutoff].drop_duplicates():
            for row in range(2):
                axes[row, column].axvline(threshold, color='#888888', linestyle=':', linewidth=.8, alpha=.65)
        for index, point in enumerate(selected.itertuples()):
            x = point.tx if direction == 'isolation' else point.ty
            axes[1, column].plot([x], [.025 + .033 * index], marker=point.plot_marker,
                                 color=point.plot_color, markersize=6.4, linestyle='none',
                                 transform=axes[1, column].get_xaxis_transform(), zorder=8)
        axes[1, column].set_xlabel('Mu-LJ isolation' if direction == 'isolation' else '|Δφ| [rad]', fontsize=11)
        for axis in axes[:, column]:
            axis.set_xlim(*limits)
            axis.grid(alpha=.2, linewidth=.7)
            axis.tick_params(labelsize=10)
    axes[0, 0].set_ylabel('Expected events per native bin\n(symlog; ±√sumw²)', fontsize=10)
    axes[1, 0].set_ylabel('Fraction per native bin\n(full conditional normalization)', fontsize=10)
    handles = [finalize_local_scan_plots_Line2D([0], [0], color=point.plot_color, marker=point.plot_marker,
                      linestyle='none', markersize=7, label=finalize_local_scan_plots__scan_wp_label(point))
               for point in selected.itertuples()]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .068), ncol=3,
               fontsize=9.1, frameon=False, columnspacing=1.7, handletextpad=.5)
    fig.text(.5, .015, 'Shown ranges: isolation 0–0.5 and |Δφ| 2.2–π. Shape denominators use the complete conditional domain, including content outside the shown ranges.\n'
             'Isolation underflow is excluded; overflow is folded into the final [2,5] native bin. Angle flows are folded into their boundary bins.\n'
             'Shared conditional curves are drawn once. Dotted lines and colored marker rows mark the highlighted WP cuts; normalized shapes have no statistical error band.',
             ha='center', va='bottom', fontsize=8.7)
    fig.subplots_adjust(left=.075, right=.985, top=.865, bottom=.22, hspace=.18, wspace=.18)
    return fig


# Additional notebook views: abcd07_stats_fragment
import numpy as finalize_local_scan_plots_np
import pandas as finalize_local_scan_plots_pd
import matplotlib.pyplot as finalize_local_scan_plots_plt
from matplotlib.lines import Line2D as finalize_local_scan_plots_Line2D
from matplotlib.patches import Patch as finalize_local_scan_plots_Patch; from matplotlib.patches import Rectangle as finalize_local_scan_plots_Rectangle


def finalize_local_scan_plots__statplot_points(result):
    return result['tables']['working_points'].sort_values('wp').reset_index(drop=True)


def finalize_local_scan_plots__statplot_labels(points, cuts=False):
    return [f'{row.wp}  {row.tx:.2f}/{row.ty:.1f}' if cuts else row.wp
            for row in points.itertuples()]


def finalize_local_scan_plots__statplot_ticks(ax, points, cuts=False):
    ax.set_xticks(finalize_local_scan_plots_np.arange(len(points)), finalize_local_scan_plots__statplot_labels(points, cuts),
                  rotation=70 if cuts else 0, ha='right' if cuts else 'center', fontsize=10)
    for tick, point in zip(ax.get_xticklabels(), points.itertuples()):
        if point.is_reference:
            tick.set_color('#6C2945')
            tick.set_fontweight('bold')
        elif point.is_top5:
            tick.set_color('#B65E00')
            tick.set_fontweight('bold')


def finalize_local_scan_plots__statplot_highlight(ax, points):
    for index, point in enumerate(points.itertuples()):
        if point.is_top5:
            ax.axvspan(index - .43, index + .43, color='#F5B65A', alpha=.12, zorder=0)
        if point.is_reference:
            ax.axvline(index, color='#6C2945', ls=':', lw=1.4, alpha=.7, zorder=0)


def finalize_local_scan_plots__statplot_note(fig, text):
    fig.text(.055, .025, text, fontsize=10.3, va='bottom', color='#404040', linespacing=1.35)


def finalize_local_scan_plots__local_toy_rankings(result, config):
    """Compare correlated-bin replica frequencies without confidence claims."""
    points = finalize_local_scan_plots__statplot_points(result)
    table = result['tables'].get('ranking_toys', finalize_local_scan_plots_pd.DataFrame())
    diagnostic = result['tables'].get('toy_diagnostics', finalize_local_scan_plots_pd.DataFrame())
    fig, axes = finalize_local_scan_plots_plt.subplots(2, 2, figsize=(16, 9), sharex='col')
    fig.suptitle('2mu2e | Is the local WP preference stable under shared MC fluctuations?',
                 x=.055, y=.965, ha='left', fontsize=19, fontweight='bold')
    fig.text(.055, .917, 'Positive-bin scaled-Poisson moment replicas | all 16 overlapping WPs varied together',
             fontsize=12, color='#444444')
    models = [('background_only', 'Background fluctuations only'),
              ('background_and_signal', 'Background + signal fluctuations')]
    x = finalize_local_scan_plots_np.arange(len(points))
    for column, (model, title) in enumerate(models):
        upper, lower = axes[:, column]
        selected = table.loc[table.model.eq(model)].set_index('wp').reindex(points.wp) if len(table) else finalize_local_scan_plots_pd.DataFrame()
        info = diagnostic.loc[diagnostic.model.eq(model)].iloc[0] if len(diagnostic) else None
        valid = len(selected) and selected.p_top1.notna().any()
        if not valid:
            for ax in (upper, lower):
                ax.text(.5, .5, 'No supported moment-replica ranking\nSee toy_diagnostics.csv for the explicit status.',
                        transform=ax.transAxes, ha='center', va='center', fontsize=12)
            upper.set_title(title, fontsize=13)
            continue
        upper.bar(x - .17, selected.p_top1 * 100, width=.31, color='#D55E00', label='Best WP (ties shared)')
        upper.bar(x + .17, selected.p_top5 * 100, width=.31, color='#0072B2', label='Top five (ties shared)')
        upper.set_ylim(0, 105)
        upper.set_ylabel('Conditional replica frequency [%]', fontsize=11)
        upper.set_title(f'{title}\n{int(info.n_complete_toys):,}/{int(info.n_toys):,} complete replicas '
                        f'({100 * info.complete_fraction:.1f}%)', fontsize=12.5, pad=10)
        upper.legend(loc='upper right', fontsize=10, frameon=True, facecolor='white', framealpha=.95)
        middle = selected.ratio_median.to_numpy(float)
        low, high = selected.ratio_q10.to_numpy(float), selected.ratio_q90.to_numpy(float)
        lower.errorbar(x, middle, yerr=finalize_local_scan_plots_np.vstack([middle - low, high - middle]), fmt='o',
                       ms=4.5, color='#0072B2', elinewidth=1.2, capsize=2.5, zorder=3)
        highlighted = points.is_top5.to_numpy(bool)
        lower.scatter(x[highlighted], middle[highlighted], color='#D55E00', s=26, zorder=4)
        reference = points.is_reference.to_numpy(bool)
        lower.scatter(x[reference], middle[reference], color='#6C2945', marker='D', s=29, zorder=5)
        lower.axhline(1, ls='--', color='#555555', lw=1)
        lower.set_ylabel('Median Z(WP) / Z(reference)', fontsize=11)
        lower.set_title('Central 80% spread across complete MC replicas', fontsize=12.5, pad=7)
        lower.set_ylim(bottom=0)
        for ax in (upper, lower):
            finalize_local_scan_plots__statplot_highlight(ax, points)
            ax.grid(axis='y', alpha=.2)
            ax.set_xlim(-.65, len(points) - .35)
            ax.tick_params(axis='y', labelsize=10)
        finalize_local_scan_plots__statplot_ticks(lower, points, cuts=True)
        lower.set_xlabel('WP ID  |  isolation / |Delta phi| [rad]', fontsize=10.5)
    fig.subplots_adjust(left=.065, right=.975, top=.82, bottom=.29, hspace=.36, wspace=.19)
    finalize_local_scan_plots__statplot_note(fig, 'Ranks use the same 60 separate signal hypotheses. Frequencies are conditional diagnostics, not confidence probabilities.\n'
                       'Replicas with zero background at any WP or undefined reference Z are counted and excluded from this joint ranking.\n'
                       'Orange labels/shading: nominal MC top five. Purple: reference. Error bars are replica spread, not model-to-model spread.')
    return fig


def finalize_local_scan_plots__local_toy_covariance(result, config):
    """Show the actual background-A covariance implied by shared selections."""
    points = finalize_local_scan_plots__statplot_points(result)
    table = result['tables'].get('rank_covariance', finalize_local_scan_plots_pd.DataFrame())
    fig = finalize_local_scan_plots_plt.figure(figsize=(16, 9))
    fig.suptitle('2mu2e | Overlapping WPs share background-MC statistical fluctuations',
                 x=.055, y=.965, ha='left', fontsize=19, fontweight='bold')
    fig.text(.055, .917, 'Correlation of the nominal SR A-yield estimates; not a correlation between signal hypotheses',
             fontsize=12, color='#444444')
    ax = fig.add_axes([.13, .155, .49, .69])
    if table.empty:
        ax.text(.5, .5, 'Covariance unavailable', ha='center', va='center', transform=ax.transAxes, fontsize=13)
        return fig
    matrix = table.pivot(index='wp_a', columns='wp_b', values='correlation').reindex(index=points.wp, columns=points.wp).to_numpy(float)
    shown = ax.imshow(matrix, origin='upper', vmin=0, vmax=1, cmap='Blues', aspect='equal')
    for i in range(len(points)):
        for j in range(len(points)):
            if finalize_local_scan_plots_np.isfinite(matrix[i, j]):
                ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', fontsize=9.8,
                        color='white' if matrix[i, j] >= .66 else '#222222')
    ax.set_xticks(finalize_local_scan_plots_np.arange(len(points)), points.wp, rotation=0, fontsize=10)
    ax.set_yticks(finalize_local_scan_plots_np.arange(len(points)), [f'{r.wp}  {r.tx:.2f}/{r.ty:.1f}' for r in points.itertuples()], fontsize=10)
    ax.set_xlabel('WP ID', fontsize=11)
    ax.set_ylabel('WP ID  |  isolation / |Delta phi| [rad]', fontsize=11)
    ax.set_xticks(finalize_local_scan_plots_np.arange(-.5, len(points), 1), minor=True)
    ax.set_yticks(finalize_local_scan_plots_np.arange(-.5, len(points), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=.45)
    ax.tick_params(which='minor', bottom=False, left=False)
    for index, row in enumerate(points.itertuples()):
        if row.is_reference or row.is_top5:
            color = '#6C2945' if row.is_reference else '#D55E00'
            ax.add_patch(finalize_local_scan_plots_Rectangle((index - .5, index - .5), 1, 1, fill=False, lw=2, edgecolor=color))
            ax.get_yticklabels()[index].set_color(color)
            ax.get_yticklabels()[index].set_fontweight('bold')
    cax = fig.add_axes([.635, .155, .014, .69])
    cb = fig.colorbar(shown, cax=cax)
    cb.set_label('Correlation of SR background A yields', fontsize=11)
    cb.ax.tick_params(labelsize=10)
    info = fig.add_axes([.71, .16, .25, .68])
    info.axis('off')
    off_diagonal = matrix[~finalize_local_scan_plots_np.eye(len(points), dtype=bool)]
    info.text(0, 1, 'Shared-event covariance', fontsize=15, fontweight='bold', va='top')
    info.text(0, .89, 'For independent Poisson event fluctuations,\n'
              'the covariance is sumw2 in the common\n'
              'part of the two A selections.', fontsize=11.5, va='top', linespacing=1.6)
    info.text(0, .69, r'$A_i\cap A_j=A(\min(t_{x,i},t_{x,j}),\max(t_{y,i},t_{y,j}))$',
              fontsize=11.5, va='top')
    info.text(0, .58, f'Off-diagonal correlation\nMinimum: {finalize_local_scan_plots_np.nanmin(off_diagonal):.3f}\n'
              f'Median: {finalize_local_scan_plots_np.nanmedian(off_diagonal):.3f}\nMaximum: {finalize_local_scan_plots_np.nanmax(off_diagonal):.3f}',
              fontsize=12, va='top', linespacing=1.55)
    top = points.loc[points.is_top5].sort_values('rank')
    info.text(0, .31, 'Nominal top five\n' + '\n'.join(f'{r.wp}: iso {r.tx:.2f}, phi {r.ty:.1f}' for r in top.itertuples()),
              fontsize=11.5, va='top', linespacing=1.5)
    finalize_local_scan_plots__statplot_note(fig, 'Covariance uses stored sumw2 in intersecting selections. It is not obtained by independently fluctuating each WP.\n'
                       'Orange diagonal outlines: nominal top five. Purple outline: reference. No nominal DATA is used.')
    return fig


def finalize_local_scan_plots__local_likelihood_figure(result, config):
    """Show declared likelihood scenarios or concrete MC inputs pending a choice."""
    points = finalize_local_scan_plots__statplot_points(result)
    table = result['tables'].get('likelihood_summary', finalize_local_scan_plots_pd.DataFrame())
    fig = finalize_local_scan_plots_plt.figure(figsize=(16, 9))
    x = finalize_local_scan_plots_np.arange(len(points))
    if not table.empty:
        fig.suptitle('2mu2e | Conditional four-region Asimov sensitivity scenarios',
                     x=.055, y=.965, ha='left', fontsize=19, fontweight='bold')
        fig.text(.055, .917, 'MC background truth | signal enters all ABCD bins | declared log-kappa constraints',
                 fontsize=12, color='#444444')
        axes = fig.subplots(1, 2)
        widths = sorted(table.kappa_log_width.unique())
        colors = finalize_local_scan_plots_plt.cm.viridis(finalize_local_scan_plots_np.linspace(.12, .9, max(2, len(widths))))
        for color, width in zip(colors, widths):
            group = table.loc[table.kappa_log_width.eq(width)].set_index('wp').reindex(points.wp)
            label = 'Fixed MC kappa' if width == 0 else ('Free kappa' if finalize_local_scan_plots_np.isinf(width) else f'kappa up factor {finalize_local_scan_plots_np.exp(width):.2f}')
            axes[0].plot(x, group.z_abcd_median, 'o-', color=color, ms=4, lw=1.2, label=label)
            axes[1].plot(x, group.z_abcd_ratio_median, 'o-', color=color, ms=4, lw=1.2, label=label)
            axes[1].fill_between(x, group.z_abcd_ratio_q10.to_numpy(float), group.z_abcd_ratio_q90.to_numpy(float), color=color, alpha=.1)
        axes[0].set_ylabel('Median conditional Asimov Z', fontsize=12)
        axes[1].set_ylabel('Median Z(WP) / Z(reference)', fontsize=12)
        axes[0].set_title('Absolute model-by-model sensitivity', fontsize=13)
        axes[1].set_title('Relative sensitivity and 10–90% model spread', fontsize=13)
        axes[1].axhline(1, color='#555555', ls='--', lw=1)
        for ax in axes:
            finalize_local_scan_plots__statplot_ticks(ax, points, cuts=True)
            finalize_local_scan_plots__statplot_highlight(ax, points)
            ax.set_xlim(-.5, len(points) - .5)
            ax.grid(alpha=.2)
            ax.tick_params(axis='y', labelsize=10)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.52, .875), ncol=min(4, len(labels)), fontsize=11, frameon=False)
        fig.subplots_adjust(left=.07, right=.975, top=.78, bottom=.245, wspace=.2)
        finalize_local_scan_plots__statplot_note(fig, 'These are conditional MC-Asimov scenarios, not a final discovery significance or a calibrated VR-to-SR transfer.\n'
                           'Sideband Poisson statistics and signal leakage are included. Background/signal MC statistical nuisances are not included in this likelihood.\n'
                           'Intervals show spread across separate signal hypotheses. Invalid or zero-reference ratios stay undefined.')
        return fig
    fig.suptitle('2mu2e | Four-region likelihood: inputs prepared, kappa assumption pending',
                 x=.055, y=.965, ha='left', fontsize=18.5, fontweight='bold')
    fig.text(.055, .917, 'NOT RUN | No VR-to-SR kappa constraint has been adopted | nominal DATA remains blind',
             fontsize=12, color='#9A4A00')
    inputs = result['tables'].get('likelihood_inputs', finalize_local_scan_plots_pd.DataFrame())
    grid = fig.add_gridspec(2, 2, left=.07, right=.975, top=.84, bottom=.12,
                           height_ratios=[1.28, 1.], wspace=.22, hspace=.48)
    kappa_ax, neff_ax = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    left, right = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    if inputs.empty:
        kappa_ax.text(.5, .5, 'MC input table unavailable', ha='center', va='center', transform=kappa_ax.transAxes, fontsize=12)
        return fig
    one = inputs.drop_duplicates('wp').set_index('wp').reindex(points.wp)
    kappa_ax.errorbar(x, one.kappa_mc, yerr=one.kappa_mc_error, fmt='o', color='#0072B2',
                     capsize=3, ms=4.5, label='MC central value and propagated MC error')
    kappa_ax.axhspan(.9, 1.1, color='#56A66B', alpha=.13)
    kappa_ax.axhline(1, color='#555555', ls='--', lw=1)
    kappa_ax.set_title('Nominal SR kappa diagnostic', fontsize=13)
    kappa_ax.set_ylabel('kappa_MC ± propagated error', fontsize=11)
    kappa_ax.legend(fontsize=10, loc='upper left', frameon=False)
    neff_ax.plot(x, one.min_sideband_neff, 'o-', color='#0072B2', ms=4.5, lw=1.1, label='Minimum of B/C/D N_eff')
    neff_ax.plot(x, one.background_neff_A, 's-', color='#D55E00', ms=4, lw=1.1, label='A background N_eff')
    neff_ax.axhline(10, color='#666666', ls='--', lw=1, label='N_eff=10 diagnostic')
    neff_ax.set_ylim(0, max(11., float(one.background_neff_A.max()) * 1.2))
    neff_ax.set_title('Effective background-MC support', fontsize=13)
    neff_ax.set_ylabel('Effective MC count', fontsize=11)
    neff_ax.legend(fontsize=10, loc='upper right', frameon=False)
    for ax in (kappa_ax, neff_ax):
        finalize_local_scan_plots__statplot_ticks(ax, points)
        finalize_local_scan_plots__statplot_highlight(ax, points)
        ax.set_xlim(-.5, len(points) - .5)
        ax.grid(axis='y', alpha=.2)
        ax.tick_params(axis='y', labelsize=10)
        ax.set_xlabel('WP ID (orange: nominal top five; purple: reference)', fontsize=10)
    left.axis('off')
    left.text(0, .98, 'Inputs are concrete; no fit assumption is hidden', fontsize=14, fontweight='bold', va='top')
    valid = int(inputs.input_valid.sum())
    left.text(0, .77, f'{len(inputs):,} MC-only input rows = 16 WPs × 60 signal hypotheses\n'
              f'{valid:,}/{len(inputs):,} rows have positive background and nonnegative signal in every ABCD bin.\n'
              'Signal and background yields, sumw2, kappa_MC and support flags\n'
              'are exported in likelihood_inputs.csv. No nominal DATA is read.',
              fontsize=11.5, va='top', linespacing=1.65)
    right.axis('off')
    right.text(0, .98, 'Proposed conditional model and pending choice', fontsize=14, fontweight='bold', va='top')
    right.text(0, .75, r'$n_r=s_r+b_r,\quad b_A=\kappa\,b_B b_C/b_D$', fontsize=15, va='top')
    right.text(0, .51, 'A kappa center and its constraint need an explicit justification.\n'
               'The displayed MC kappa error is not adopted as that constraint.\n'
               'Sideband statistics and signal leakage will enter the joint model.\n'
               'The current input support does not establish VR-to-SR transfer.',
               fontsize=11.5, va='top', linespacing=1.65)
    finalize_local_scan_plots__statplot_note(fig, 'This page prepares step 4 without choosing a closure correction or uncertainty. No numerical likelihood ranking is reported.\n'
                       'MC central-value kappa and low effective counts are diagnostics; all nominal DATA remains blind.')
    return fig


def finalize_local_scan_plots__vr2focus_points(result):
    """Keep exactly five selected WPs; style a reference only if it is selected."""
    points = result['tables']['working_points'].sort_values('rank').copy()
    if len(points) != 5 or points.is_reference.sum() > 1 or points.wp.duplicated().any():
        raise ValueError('The focused notebook requires exactly five unique selected WPs.')
    palette = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00')
    shapes = ('o', 's', '^', 'D', 'P')
    alternatives = 0
    colors, markers, legends = [], [], []
    for point in points.itertuples():
        if point.is_reference:
            colors.append('#666666')
            markers.append('X')
            name = 'Ref'
        else:
            colors.append(palette[alternatives])
            markers.append(shapes[alternatives])
            name = point.wp
            alternatives += 1
        legends.append(f'{name}: iso={point.tx:.2f}, |Δφ|={point.ty:.1f}')
    points['focus_color'] = colors
    points['focus_marker'] = markers
    points['focus_legend'] = legends
    return points


def finalize_local_scan_plots__vr2focus_rows(result, table, scope, source=None):
    frame = result['tables'][table]
    chosen = frame.loc[frame.scope.eq(scope)]
    if source is not None:
        chosen = chosen.loc[chosen.source.eq(source)]
    return chosen.copy()


def finalize_local_scan_plots__vr2focus_edges(values, low, high):
    """Midpoint display cells for nonuniform threshold coordinates; no interpolation."""
    values = finalize_local_scan_plots_np.asarray(values, float)
    if values.ndim != 1 or len(values) < 2 or finalize_local_scan_plots_np.any(finalize_local_scan_plots_np.diff(values) <= 0):
        raise ValueError('Threshold coordinates must be unique and strictly increasing.')
    if values[0] < low or values[-1] > high + 1e-12:
        raise ValueError('Threshold coordinates are outside the requested numeric display range.')
    return finalize_local_scan_plots_np.r_[low, (values[:-1] + values[1:]) / 2., high]


def finalize_local_scan_plots__vr2focus_legend(fig, points, y=.105):
    handles = [finalize_local_scan_plots_Line2D([0], [0], marker=point.focus_marker, color=point.focus_color,
                      linestyle='None', markersize=8, markeredgecolor='white',
                      markeredgewidth=.7, label=point.focus_legend)
               for point in points.itertuples()]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, y),
               ncol=3, frameon=False, fontsize=10, columnspacing=1.8,
               handletextpad=.5)


def finalize_local_scan_plots__vr2focus_full_map(result, config, source):
    """Label every valid native κ value on the notebook-01–06 threshold grid."""
    scopes = ('VR2_PP', 'VR2_PD', 'VR2_DP')
    fig, axes = finalize_local_scan_plots_plt.subplots(1, 3, figsize=(16, 9))
    topology_label = '4μ' if config['topology'] == '4mu' else '2μ2e'
    fig.suptitle(f'{topology_label} · Intermediate MJJ VR · full-range κ maps ({source})', x=.065, y=.965,
                 ha='left', fontsize=20, fontweight='bold')
    fig.text(.065, .916, '150 ≤ MJJ < 300 GeV; 0 ≤ partner isolation < 0.5; PP and PD evaluate the candidates, DP provides a cross-check.',
             fontsize=11.5, color='#444444')
    norm = finalize_local_scan_plots_TwoSlopeNorm(vmin=0., vcenter=1., vmax=2.)
    cmap = finalize_local_scan_plots_plt.get_cmap('coolwarm').copy()
    cmap.set_bad('#e1e1e1')
    image = None
    for ax, scope in zip(axes, scopes):
        frame = finalize_local_scan_plots__vr2focus_rows(result, 'full_closure', scope, source)
        if len(frame) != 154 or frame.duplicated(['tx', 'ty']).any():
            raise AssertionError(f'{scope}/{source}: expected the complete 154-point threshold grid.')
        xs, ys = finalize_local_scan_plots_np.sort(frame.tx.unique()), finalize_local_scan_plots_np.sort(frame.ty.unique())
        if len(xs) != 11 or len(ys) != 14:
            raise AssertionError('The full threshold grid must contain 11 isolation and 14 angle boundaries.')
        values = frame.pivot(index='ty', columns='tx', values='kappa').reindex(index=ys, columns=xs).to_numpy(float)
        if source == 'DATA':
            blocked = frame.loc[~frame.approved]
            if not blocked[['A', 'pred', 'kappa']].isna().all().all():
                raise AssertionError('The full κ map must retain all existing DATA gate masks.')
        image = ax.imshow(finalize_local_scan_plots_np.ma.masked_invalid(values), origin='lower', aspect='auto',
                          interpolation='nearest', norm=norm, cmap=cmap)
        for row in frame.itertuples():
            ix = int(finalize_local_scan_plots_np.flatnonzero(finalize_local_scan_plots_np.isclose(xs, row.tx, rtol=0, atol=1e-12))[0])
            iy = int(finalize_local_scan_plots_np.flatnonzero(finalize_local_scan_plots_np.isclose(ys, row.ty, rtol=0, atol=1e-12))[0])
            if source == 'DATA' and not bool(row.approved):
                ax.add_patch(finalize_local_scan_plots_Rectangle((ix-.5, iy-.5), 1., 1., facecolor='none',
                                       edgecolor='#aaaaaa', hatch='///', linewidth=.25))
            elif not finalize_local_scan_plots_np.isfinite(row.kappa):
                ax.text(ix, iy, '×', color='#777777', ha='center', va='center', fontsize=9)
            else:
                value = float(row.kappa)
                label = f'{value:.1e}' if abs(value) >= 100 else f'{value:.2f}'
                color = cmap(norm(value))
                luminance = .299*color[0] + .587*color[1] + .114*color[2]
                ax.text(ix, iy, label, ha='center', va='center', fontsize=7.1,
                        color='white' if luminance < .48 else '#222222')
        local_x = finalize_local_scan_plots_np.flatnonzero((xs >= .20-1e-12) & (xs <= .40+1e-12))
        local_y = finalize_local_scan_plots_np.flatnonzero((ys >= 2.4-1e-12) & (ys <= 2.8+1e-12))
        if len(local_x) != 4 or len(local_y) != 4:
            raise AssertionError('The local highlight must enclose exactly the 4×4 native candidate grid.')
        ax.add_patch(finalize_local_scan_plots_Rectangle((local_x[0]-.5, local_y[0]-.5), len(local_x), len(local_y), fill=False,
                               edgecolor='#762A83', linewidth=2.5, zorder=7))
        ax.set(xlim=(-.5, len(xs)-.5), ylim=(-.5, len(ys)-.5),
               xticks=finalize_local_scan_plots_np.arange(len(xs)), xticklabels=[f'{x:g}' for x in xs],
               yticks=finalize_local_scan_plots_np.arange(len(ys)),
               yticklabels=['π' if finalize_local_scan_plots_np.isclose(y, finalize_local_scan_plots_np.pi) else f'{y:g}' for y in ys],
               xlabel='Leading mu-LJ isolation boundary' if config['topology'] == '4mu' else 'Mu isolation boundary',
               ylabel='|Δφ| boundary [rad]',
               title=scope.replace('VR2_', ''))
        ax.tick_params(labelsize=9)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(False)
    fig.subplots_adjust(left=.065, right=.91, top=.83, bottom=.18, wspace=.31)
    color_axis = fig.add_axes([.932, .18, .014, .65])
    fig.colorbar(image, cax=color_axis, extend='both', label='κ = A / (BC/D)')
    fig.text(.5, .025, 'Equal-spaced native threshold cells, as in notebooks 01–06; tick labels give the actual thresholds. Zero splits are excluded.\n'
             'Numbers are κ central values. Purple box: iso 0.20–0.40 and |Δφ| 2.4–2.8. Colors saturate outside 0–2; errors remain in the tables.\n'
             + ('Hatched cells are DATA-gate blocked; gray cells / × are undefined. Gate approval is not a closure-quality criterion.'
                if source == 'DATA' else 'Gray cells / × are undefined. MC κ can be poorly constrained by low effective statistics.'),
             ha='center', va='bottom', fontsize=9, color='#444444')
    return fig


def finalize_local_scan_plots__vr2focus_joint(result, config):
    """Compare PP and PD closure jointly without defining a new combined score."""
    points = finalize_local_scan_plots__vr2focus_points(result)
    pp = finalize_local_scan_plots__vr2focus_rows(result, 'closure', 'VR2_PP', 'DATA').set_index('wp')
    pd_rows = finalize_local_scan_plots__vr2focus_rows(result, 'closure', 'VR2_PD', 'DATA').set_index('wp')
    fig, ax = finalize_local_scan_plots_plt.subplots(figsize=(16, 9))
    topology_label = '4μ' if config['topology'] == '4mu' else '2μ2e'
    fig.suptitle(f'{topology_label} · Intermediate MJJ VR · PP–PD joint closure view', x=.075, y=.965,
                 ha='left', fontsize=20, fontweight='bold')
    fig.text(.075, .916, '150 ≤ MJJ < 300 GeV. Each marker is one selected WP; PP and PD remain separate validation measurements.',
             fontsize=12, color='#444444')
    ax.axvspan(.9, 1.1, color='#83B881', alpha=.12)
    ax.axhspan(.9, 1.1, color='#83B881', alpha=.12)
    ax.axvline(1., color='#5F775B', ls='--', lw=1)
    ax.axhline(1., color='#5F775B', ls='--', lw=1)
    ax.plot(1., 1., '+', color='black', ms=13, mew=1.8)
    xbounds, ybounds = [.85, 1.15], [.85, 1.15]
    for point in points.itertuples():
        first, second = pp.loc[point.wp], pd_rows.loc[point.wp]
        if not bool(first.approved) or not bool(second.approved):
            continue
        if not finalize_local_scan_plots_np.isfinite([first.kappa, second.kappa, first.kappa_error, second.kappa_error]).all():
            continue
        ax.errorbar(first.kappa, second.kappa, xerr=first.kappa_error, yerr=second.kappa_error,
                    color=point.focus_color, fmt=point.focus_marker, markersize=10,
                    capsize=5, elinewidth=1.3, markeredgecolor='white', markeredgewidth=.8,
                    alpha=.9, zorder=5)
        ax.annotate(point.wp, (first.kappa, second.kappa), xytext=(8, 7 if point.is_reference else -12),
                    textcoords='offset points', fontsize=10, color=point.focus_color)
        xbounds.extend([first.kappa-first.kappa_error-.04, first.kappa+first.kappa_error+.04])
        ybounds.extend([second.kappa-second.kappa_error-.04, second.kappa+second.kappa_error+.04])
    ax.set(xlim=(min(xbounds), max(xbounds)), ylim=(min(ybounds), max(ybounds)),
           xlabel='PP DATA κ ± propagated uncertainty', ylabel='PD DATA κ ± propagated uncertainty')
    ax.tick_params(labelsize=11)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.grid(alpha=.2)
    fig.subplots_adjust(left=.10, right=.96, top=.83, bottom=.23)
    finalize_local_scan_plots__vr2focus_legend(fig, points)
    fig.text(.5, .025, 'The cross at (1,1) is exact closure; shaded ±10% bands are visual guides.\n'
             'Error bars are marginal uncertainties. Different WPs reuse events; this is not a joint confidence region or an independent-WP test.',
             ha='center', va='bottom', fontsize=10, color='#444444')
    return fig


def finalize_local_scan_plots__vr2focus_summary(result, config):
    """Keep yields, closure and contamination together for exactly five WPs."""
    points = finalize_local_scan_plots__vr2focus_points(result)
    scopes = ('VR2_PP', 'VR2_PD', 'VR2_DP')
    fig, axes = finalize_local_scan_plots_plt.subplots(3, 3, figsize=(16, 9), sharex='col',
                             gridspec_kw={'height_ratios': [1.2, 1., .9]})
    topology_label = '4μ' if config['topology'] == '4mu' else '2μ2e'
    fig.suptitle(f'{topology_label} · Intermediate MJJ VR · selected WP comparison', x=.065, y=.97,
                 ha='left', fontsize=20, fontweight='bold')
    fig.text(.065, .925, '150 ≤ MJJ < 300 GeV. Five MC-selected WPs; PP and PD assess the choice, with DP shown alongside.',
             fontsize=11.5, color='#444444')
    x = finalize_local_scan_plots_np.arange(len(points))
    contamination_values = []
    for scope in scopes:
        frame = finalize_local_scan_plots__vr2focus_rows(result, 'closure', scope, 'MC').set_index('wp').reindex(points.wp)
        contamination_values.extend(frame.max_S_B.to_numpy(float))
    positive = finalize_local_scan_plots_np.asarray(contamination_values, float)
    positive = positive[finalize_local_scan_plots_np.isfinite(positive) & (positive > 0)]
    minimum = min(.02, positive.min()*.55) if len(positive) else 1e-4
    maximum = max(.17, positive.max()*1.6) if len(positive) else .17
    for column, scope in enumerate(scopes):
        data = finalize_local_scan_plots__vr2focus_rows(result, 'closure', scope, 'DATA').set_index('wp').reindex(points.wp)
        mc = finalize_local_scan_plots__vr2focus_rows(result, 'closure', scope, 'MC').set_index('wp').reindex(points.wp)
        if not data.loc[~data.approved, ['A', 'pred', 'kappa']].isna().all().all():
            raise AssertionError('Selected DATA quantities must retain their contamination gate.')
        axes[0, column].errorbar(x-.07, data.A, yerr=data.A_error, color='#222222',
                                 marker='o', linestyle='none', ms=5, capsize=3, label='Observed A')
        axes[0, column].errorbar(x+.07, data.pred, yerr=data.pred_error, color='#D55E00',
                                 marker='s', markerfacecolor='white', linestyle='none',
                                 ms=5, capsize=3, label='ABCD BC/D')
        axes[0, column].set_title(scope.replace('VR2_', ''), fontsize=14, pad=8)
        axes[0, column].set_ylim(bottom=0)
        axes[1, column].axhspan(.9, 1.1, color='#83B881', alpha=.17)
        axes[1, column].axhline(1., color='#5F775B', ls='--', lw=1)
        axes[2, column].axhline(.1, color='#B52B25', ls='--', lw=1)
        axes[2, column].set_yscale('log')
        axes[2, column].set_ylim(minimum, maximum)
        for j, point in enumerate(points.itertuples()):
            row = data.loc[point.wp]
            if bool(row.approved) and finalize_local_scan_plots_np.isfinite(row.kappa):
                axes[1, column].errorbar(j, row.kappa, yerr=row.kappa_error,
                                         fmt=point.focus_marker, color=point.focus_color,
                                         markersize=7, capsize=3, markeredgecolor='white', markeredgewidth=.6)
            else:
                axes[1, column].text(j, .05, 'gated / undefined', rotation=90,
                                         transform=axes[1, column].get_xaxis_transform(),
                                         ha='center', va='bottom', fontsize=8, color='#777777')
            value = mc.loc[point.wp].max_S_B
            if finalize_local_scan_plots_np.isfinite(value) and value > 0:
                axes[2, column].scatter(j, value, color=point.focus_color, marker=point.focus_marker,
                                        s=48, edgecolors='white', linewidth=.7, zorder=5)
        for axis in axes[:, column]:
            axis.set_xlim(-.4, len(points)-.6)
            axis.grid(alpha=.2)
            axis.tick_params(labelsize=10)
        axes[2, column].set_xticks(x, [f'{r.wp}\n{r.tx:.2f}, {r.ty:.1f}' for r in points.itertuples()], fontsize=9)
        axes[2, column].set_xlabel('WP · (iso, |Δφ|)', fontsize=10)
    axes[0, 0].set_ylabel('DATA events\nA and BC/D', fontsize=11)
    axes[1, 0].set_ylabel('DATA κ ± uncertainty', fontsize=11)
    axes[2, 0].set_ylabel('Maximum MC S/B\n120 hypotheses × ABCD', fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(.965, .925),
               ncol=2, frameon=False, fontsize=10)
    fig.subplots_adjust(left=.075, right=.98, top=.85, bottom=.17, hspace=.20, wspace=.25)
    fig.text(.5, .025, 'Top two rows: gated VR DATA and propagated uncertainties. Bottom row: MC contamination, with the same log scale in PP/PD/DP.\n'
             'The dashed 10% contamination line controls DATA access; the green κ band is only a closure reference. Nearby WPs share events.',
             ha='center', va='bottom', fontsize=9.5, color='#444444')
    return fig


def finalize_local_scan_plots_vr2_focus_figure_definitions():
    """The focused notebook contains eight figures in a fixed, compact order."""
    entries = [
        ('significance_comparison', 'Selected-WP counting-metric comparison', 'S/sqrt(B), S/sqrt(S+B), and Asimov Z, with the same 60-model ranges and a shared log scale.'),
        ('full_data_kappa', 'Full-range intermediate MJJ VR DATA κ', '150 ≤ MJJ < 300 GeV; all 154 stored threshold points labeled by κ in PP, PD and DP; the local range is boxed.'),
        ('full_mc_kappa', 'Full-range intermediate MJJ VR MC κ', '150 ≤ MJJ < 300 GeV; the same annotated threshold grid, retaining sparse-MC limitations.'),
        ('pp_pd_joint', 'PP–PD joint closure view', 'Compare the five selected WPs using both validation regions without merging them.'),
        ('selected_summary', 'Five-WP PP/PD/DP summary', 'Observed A, ABCD prediction, κ and contamination at the same five boundaries.'),
        ('scan_VR2_PP', 'PP 1D stability comparisons', 'Extended isolation and angle scans with event yields, κ and MC contamination.'),
        ('scan_VR2_PD', 'PD 1D stability comparisons', 'The extended scans expose PD closure behaviour and statistical support.'),
        ('scan_VR2_DP', 'DP 1D stability comparisons', 'The lower-statistics cross-check uses the same selected boundaries and extended scan ranges.'),
    ]
    return finalize_local_scan_plots_pd.DataFrame(entries, columns=['name', 'title', 'purpose'])


def finalize_local_scan_plots_make_vr2_focus_figures(result, config):
    """Render only the full-plane overview and the five-WP VR2 comparisons."""
    finalize_local_scan_plots__vr2focus_points(result)
    if set(result['tables']['full_closure'].scope) != {'VR2_PP', 'VR2_PD', 'VR2_DP'}:
        raise ValueError('The focused figures require VR2 PP, PD and DP only.')
    with finalize_local_scan_plots_plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.titlesize': 13, 'axes.labelsize': 11,
                         'figure.max_open_warning': 0}):
        figures = [
            (finalize_local_scan_plots__vr2focus_significance(result, config), 'significance_comparison'),
            (finalize_local_scan_plots__vr2focus_full_map(result, config, 'DATA'), 'full_data_kappa'),
            (finalize_local_scan_plots__vr2focus_full_map(result, config, 'MC'), 'full_mc_kappa'),
            (finalize_local_scan_plots__vr2focus_joint(result, config), 'pp_pd_joint'),
            (finalize_local_scan_plots__vr2focus_summary(result, config), 'selected_summary'),
        ]
        figures.extend((finalize_local_scan_plots__local_scan_figure(result, config, scope), 'scan_'+scope)
                       for scope in ('VR2_PP', 'VR2_PD', 'VR2_DP'))
    if [name for _, name in figures] != finalize_local_scan_plots_vr2_focus_figure_definitions().name.tolist():
        raise AssertionError('The focused notebook figure order changed unexpectedly.')
    for figure, _ in figures:
        figure.canvas.draw()
    return figures


def finalize_local_scan_plots__vr2focus_significance(result, config):
    """Compare three absolute counting metrics for the same five WPs."""
    points = finalize_local_scan_plots__vr2focus_points(result)
    signals = result['tables']['signal_significance']
    signals = signals.loc[signals.matching_decay]
    metrics = (('s_sqrt_b', r'$S/\sqrt{B}$'),
               ('s_sqrt_s_plus_b', r'$S/\sqrt{S+B}$'),
               ('z_known_b', r'Asimov $Z_A$'))
    fig, axes = finalize_local_scan_plots_plt.subplots(1, 3, figsize=(16, 9), sharey=True)
    topology_label = '4μ' if config['topology'] == '4mu' else '2μ2e'
    fig.suptitle(f'{topology_label} · selected WP significance comparison', x=.075, y=.965,
                 ha='left', fontsize=20, fontweight='bold')
    fig.text(.075, .916, 'Nominal DD MC SR A; 60 matching signal hypotheses evaluated separately; production normalization unchanged.',
             fontsize=11.5, color='#444444')
    plotted_bounds = []
    for j, point in enumerate(points.itertuples()):
        part = signals.loc[signals.wp.eq(point.wp)]
        if len(part) != 60 or not finalize_local_scan_plots_np.isfinite(part[[column for column, _ in metrics]]).all().all():
            raise AssertionError('Each significance point needs the same 60 valid signal-model values.')
        for ax, (column, _) in zip(axes, metrics):
            lower, middle, upper = part[column].quantile([.1, .5, .9]).to_numpy(float)
            if lower <= 0:
                raise ValueError('The shared log display requires positive model-percentile bounds; no floor is applied.')
            ax.errorbar(j, middle, yerr=finalize_local_scan_plots_np.array([[middle-lower], [upper-middle]]),
                        fmt=point.focus_marker, color=point.focus_color, markersize=9,
                        capsize=5, elinewidth=1.4, markeredgecolor='white', markeredgewidth=.7)
            ax.annotate(f'{middle:.4f}',
                        (j, middle), xytext=(8, 5), textcoords='offset points', fontsize=10,
                        color=point.focus_color)
            plotted_bounds.extend([lower, upper])
    axes[0].set_yscale('log')
    axes[0].set_ylim(min(plotted_bounds)*.65, max(plotted_bounds)*1.5)
    axes[0].set_ylabel('Counting metric · median and 10–90% model range', fontsize=11)
    for ax, (_, title) in zip(axes, metrics):
        ax.set_title(title, fontsize=16)
        ax.set_xticks(finalize_local_scan_plots_np.arange(len(points)),
                      [f'{r.wp}\niso={r.tx:.2f}\n|Δφ|={r.ty:.1f}' for r in points.itertuples()], fontsize=9)
        ax.set_xlim(-.45, len(points)-.35)
        ax.set_xlabel('Selected working point', fontsize=11)
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(alpha=.22, which='major')
    fig.subplots_adjust(left=.075, right=.98, bottom=.25, top=.82, wspace=.14)
    fig.text(.5, .025,
             'Markers are medians; bars span the signal-hypothesis 10th–90th percentiles, not statistical uncertainties or confidence intervals.\n'
             'All three panels show absolute values on one shared log scale, using the same S and B for each of the 60 hypotheses.\n'
             'The five WPs retain their original selection order. These MC counting metrics do not include the full ABCD background-uncertainty model.',
             ha='center', va='bottom', fontsize=9.5, color='#444444')
    return fig


# ========================================================================
# abcd_comprehensive_plots: original study API
# ========================================================================
"""Post-discussion figures for the comprehensive 2mu2e notebook.

Consume validated tables and already selected planes. This module never opens
DATA, changes the working-point selection, or estimates a nominal DATA yield.
"""
import textwrap as finalize_comprehensive_plots_textwrap

import matplotlib.pyplot as finalize_comprehensive_plots_plt
from matplotlib.colors import LogNorm as finalize_comprehensive_plots_LogNorm
from matplotlib.lines import Line2D as finalize_comprehensive_plots_Line2D
from matplotlib.patches import Patch as finalize_comprehensive_plots_Patch; from matplotlib.patches import Rectangle as finalize_comprehensive_plots_Rectangle
import numpy as finalize_comprehensive_plots_np
import pandas as finalize_comprehensive_plots_pd

finalize_comprehensive_plots_reference_plots = _FINALIZE_NAMESPACES['abcd_finalize_plots']
finalize_comprehensive_plots_local_plots = _FINALIZE_NAMESPACES['abcd_local_scan_plots']


finalize_comprehensive_plots_PLANES = ('PP', 'PD', 'DP')
finalize_comprehensive_plots_REGION_COLORS = {'A': '#222222', 'B': '#009E73', 'C': '#D55E00', 'D': '#CC79A7'}
finalize_comprehensive_plots_MASS_COLORS = {'low': '#0072B2', 'intermediate': '#E69F00', 'high': '#CC79A7'}
finalize_comprehensive_plots_MASS_LABELS = {'low': '0–150 GeV', 'intermediate': '150–300 GeV', 'high': '≥300 GeV'}


def finalize_comprehensive_plots__table(result, name):
    frame = result['tables'][name]
    if not isinstance(frame, finalize_comprehensive_plots_pd.DataFrame):
        raise TypeError(f'{name} must be a DataFrame.')
    return frame.copy()


def finalize_comprehensive_plots__points(result):
    points = finalize_comprehensive_plots__table(result, 'working_points')
    if points.wp.duplicated().any() or points.duplicated(['tx', 'ty']).any():
        raise ValueError('Comprehensive working points must be unique.')
    return points.reset_index(drop=True)


def finalize_comprehensive_plots__labels(points):
    return [f'{point.wp}\n({point.tx:g}, {point.ty:g})' for point in points.itertuples()]


def finalize_comprehensive_plots__canvas(title, rows=1, cols=3, height=8.3, bottom=.18, top=.84):
    fig, axes = finalize_comprehensive_plots_plt.subplots(rows, cols, figsize=(16, height), squeeze=False)
    fig.suptitle('2μ2e · ' + title, x=.065, y=.97, ha='left', fontsize=19, fontweight='bold')
    fig.subplots_adjust(left=.075, right=.97, top=top, bottom=bottom, wspace=.28, hspace=.30)
    for ax in axes.ravel():
        ax.grid(alpha=.20)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)
    return fig, axes


def finalize_comprehensive_plots__caption(fig, text):
    wrapped = '\n'.join(finalize_comprehensive_plots_textwrap.fill(line, width=173) for line in text.splitlines())
    fig.text(.5, .018, wrapped, ha='center', va='bottom', fontsize=9, color='#444444')


def finalize_comprehensive_plots__cut_rows(frame, point):
    return frame.loc[finalize_comprehensive_plots_np.isclose(frame.tx, point.tx, rtol=0, atol=1e-12)
                     & finalize_comprehensive_plots_np.isclose(frame.ty, point.ty, rtol=0, atol=1e-12)]


def finalize_comprehensive_plots__positive_line(ax, x, y, **kwargs):
    values = finalize_comprehensive_plots_np.asarray(y, float)
    ax.plot(x, finalize_comprehensive_plots_np.where(finalize_comprehensive_plots_np.isfinite(values) & (values > 0), values, finalize_comprehensive_plots_np.nan), **kwargs)


def finalize_comprehensive_plots__closure(ax, x, frame, color, label, marker='o', shift=0.):
    values = frame.kappa.to_numpy(float)
    errors = frame.kappa_error.to_numpy(float)
    finalize_comprehensive_plots_reference_plots._opt_reference_errorbar(ax, finalize_comprehensive_plots_np.asarray(x) + shift, values, error=errors,
                                            fmt=marker, color=color, capsize=3, label=label)
    finalize_comprehensive_plots_reference_plots._opt_closure_limits(ax, finalize_comprehensive_plots_np.asarray(x) + shift, values, errors,
                                      note_y={'#222222': .94, '#E69F00': .84, '#0072B2': .74}.get(color, .84))


def finalize_comprehensive_plots__sb_regions(result):
    frame = finalize_comprehensive_plots__table(result, 'contamination_regions')
    frame = frame.loc[finalize_comprehensive_plots_np.isclose(frame.tx, .2) & frame.ty.between(1., 3.)]
    fig, axes = finalize_comprehensive_plots__canvas('Why does the maximum S/B rise at lower |Δφ|?', height=7.3, bottom=.22)
    for ax, plane in zip(axes[0], finalize_comprehensive_plots_PLANES):
        part = frame.loc[frame.plane.eq(plane)]
        for region in 'ABCD':
            rows = part.loc[part.region.eq(region)].sort_values('ty')
            finalize_comprehensive_plots__positive_line(ax, rows.ty, rows.max_S_B, color=finalize_comprehensive_plots_REGION_COLORS[region],
                           marker='o', ms=4, label=region)
        envelope = part.groupby('ty', sort=True).max_S_B.max()
        finalize_comprehensive_plots__positive_line(ax, envelope.index, envelope, color='#333333', ls='--', lw=1.6,
                       label='Maximum over A/B/C/D')
        ax.axhline(.1, color='#B52B25', ls=':', lw=1.2)
        ax.set(yscale='log', xlabel='|Δφ| boundary [rad]', title=f'Intermediate MJJ VR · {plane}', xlim=(1., 3.))
        ax.set_xticks([1., 1.5, 2., 2.4, 2.6, 2.8, 3.])
    axes[0, 0].set_ylabel('Maximum S/B_MC over 120 signal hypotheses')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc='lower center', bbox_to_anchor=(.5, .125), frameon=False)
    finalize_comprehensive_plots__caption(fig, 'Isolation is fixed at 0.20; 150 ≤ MJJ < 300 GeV; partner isolation < 0.5. The denominator is the direct background MC yield in each region, not BC/D.\n'
             'Lowering the angle boundary expands A/B and contracts C/D. A rising ratio must be checked against the separate signal/background yields; the maximizing signal model can change.')
    return fig


def finalize_comprehensive_plots__sb_decomposition(result):
    frame = finalize_comprehensive_plots__table(result, 'contamination_fixed')
    frame = frame.loc[finalize_comprehensive_plots_np.isclose(frame.tx, .2) & frame.region.eq('C') & frame.ty.between(1., 3.)]
    fig, axes = finalize_comprehensive_plots__canvas('Region C: shrinking denominator versus a changing envelope', rows=3,
                        height=10.1, bottom=.19, top=.85)
    for column, plane in enumerate(finalize_comprehensive_plots_PLANES):
        part = frame.loc[frame.plane.eq(plane)].sort_values('ty')
        if part.ty.duplicated().any():
            raise ValueError('One fixed reference signal is required per plane/region.')
        finalize_comprehensive_plots__positive_line(axes[0, column], part.ty, part.signal_retention, color='#0072B2', marker='o', ms=3.8, label='Fixed-model signal C')
        finalize_comprehensive_plots__positive_line(axes[0, column], part.ty, part.background_retention, color='#E69F00', marker='s', ms=3.8, label='Background MC C')
        axes[0, column].axhline(1, color='#777777', ls=':', lw=.9)
        axes[0, column].set(yscale='log', title=plane)
        finalize_comprehensive_plots__positive_line(axes[1, column], part.ty, part.max_S_B, color='#222222', marker='o', ms=3.8,
                       label='Maximum over 120 models in C')
        finalize_comprehensive_plots__positive_line(axes[1, column], part.ty, part.fixed_S_B, color='#0072B2', marker='s', ms=3.8,
                       ls='--', label='Same fixed signal model')
        axes[1, column].axhline(.1, color='#B52B25', ls=':', lw=1.)
        axes[1, column].set_yscale('log')
        axes[2, column].plot(part.ty, part.neff_background, color='#E69F00', marker='o', ms=3.8)
        axes[2, column].axhline(10, color='#888888', ls=':', lw=.9)
        axes[2, column].set(xlabel='|Δφ| boundary [rad]', ylim=(0, None))
        for ax in axes[:, column]:
            ax.set_xlim(1., 3.)
            ax.set_xticks([1., 1.5, 2., 2.4, 2.6, 2.8, 3.])
        if len(part):
            axes[0, column].text(.03, .97, str(part.iloc[0]['sample']), transform=axes[0, column].transAxes,
                                 va='top', fontsize=7.3, bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
    axes[0, 0].set_ylabel('Yield / yield at |Δφ|=2.6')
    axes[1, 0].set_ylabel('S_C / background MC C')
    axes[2, 0].set_ylabel('Background C effective MC count')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles2, labels2 = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles + handles2, labels + labels2, loc='lower center', bbox_to_anchor=(.5, .095),
               ncol=2, frameon=False, fontsize=9)
    finalize_comprehensive_plots__caption(fig, 'The fixed model is the worst C-contamination hypothesis at (iso, |Δφ|)=(0.20, 2.6), held unchanged throughout each curve.\n'
             'The upper ratios are central-value retention diagnostics, not independent statistical ratios. The lower effective counts expose sparse MC support; unsupported ratios remain gaps.')
    return fig


def finalize_comprehensive_plots__nominal_retention(result):
    points = finalize_comprehensive_plots__points(result)
    summary = finalize_comprehensive_plots__table(result, 'nominal_summary').set_index('wp').loc[points.wp]
    composition = finalize_comprehensive_plots__table(result, 'nominal_composition')
    composition = composition.loc[composition.region.eq('A')]
    fig, axes = finalize_comprehensive_plots__canvas('Nominal MC SR: signal retention and background composition', cols=2, height=7.6, bottom=.25)
    x = finalize_comprehensive_plots_np.arange(len(points))
    middle = summary.signal_retention_median.to_numpy(float)
    low, high = summary.signal_retention_q10.to_numpy(float), summary.signal_retention_q90.to_numpy(float)
    axes[0, 0].errorbar(x - .06, middle, yerr=finalize_comprehensive_plots_np.array([middle - low, high - middle]),
                      fmt='o', color='#0072B2', capsize=3, label='Signal retention: median, 10–90% models')
    axes[0, 0].plot(x + .06, summary.background_retention, 's--', color='#E69F00', label='Background MC retention')
    axes[0, 0].axhline(1, color='#777777', ls=':', lw=1)
    axes[0, 0].set_ylabel('A yield / A yield at reference (0.30, 2.6)')
    axes[0, 0].legend(fontsize=8.4, loc='upper left')
    bottom = finalize_comprehensive_plots_np.zeros(len(points))
    for component, color in finalize_comprehensive_plots_local_plots.COMPONENT_COLORS.items():
        part = composition.loc[composition.component.eq(component)].groupby('wp')['yield'].sum()
        values = part.reindex(points.wp, fill_value=0).to_numpy(float)
        axes[0, 1].bar(x, values, bottom=bottom, color=color, label=component, width=.65)
        bottom += values
    axes[0, 1].errorbar(x, summary.A, yerr=summary.A_error, color='#222222', fmt='none', capsize=3,
                      label='Total background MC ± √sumw²')
    axes[0, 1].set_ylabel('Background MC in nominal A')
    axes[0, 1].legend(fontsize=8.4)
    for ax in axes[0]:
        ax.set_xticks(x, finalize_comprehensive_plots__labels(points), fontsize=8.5)
        ax.set_ylim(bottom=0)
    finalize_comprehensive_plots__caption(fig, 'Nominal DD MC only; MJJ ≥ 150 GeV; partner isolation < 0.5. Production weights are unchanged.\n'
             'Signal ranges summarize 60 matching 2Mu2E hypotheses, not statistical confidence intervals. A large reduction of weighted MC background can be driven by very few effective events.')
    return fig


def finalize_comprehensive_plots__nominal_closure(result):
    points = finalize_comprehensive_plots__points(result)
    summary = finalize_comprehensive_plots__table(result, 'nominal_summary').set_index('wp').loc[points.wp]
    fig, axes = finalize_comprehensive_plots__canvas('Nominal DD MC closure and statistical support', height=7.5, bottom=.24)
    x = finalize_comprehensive_plots_np.arange(len(points))
    axes[0, 0].errorbar(x - .07, summary.A, yerr=summary.A_error, fmt='o', color='#222222', capsize=3, label='Direct MC A')
    axes[0, 0].errorbar(x + .07, summary.pred, yerr=summary.pred_error, fmt='s', color='#D55E00', capsize=3, label='MC B×C/D')
    axes[0, 0].set_ylabel('Weighted background MC yield')
    axes[0, 0].legend(fontsize=9)
    finalize_comprehensive_plots__closure(axes[0, 1], x, summary, '#222222', 'Nominal MC κ')
    axes[0, 1].axhspan(.9, 1.1, color='#83B881', alpha=.17)
    axes[0, 1].axhline(1, color='#5F775B', ls='--', lw=1)
    axes[0, 1].set(ylabel='κ = A / (B×C/D)', ylim=(0, 2))
    axes[0, 2].plot(x, summary.neff_A, 'o-', color='#0072B2', label='A')
    if 'min_sideband_neff' in summary:
        axes[0, 2].plot(x, summary.min_sideband_neff, 's--', color='#D55E00', label='Minimum of B/C/D')
    axes[0, 2].axhline(10, color='#888888', ls=':', label='N_eff=10 diagnostic')
    axes[0, 2].set(ylabel='Effective MC count', ylim=(0, None))
    axes[0, 2].legend(fontsize=9)
    for ax in axes[0]:
        ax.set_xticks(x, finalize_comprehensive_plots__labels(points), fontsize=8.5)
    finalize_comprehensive_plots__caption(fig, 'MC-only evidence for the WP comparison. Error bars use the existing independent-region sumw² propagation, with sparse/undefined cases kept explicit.\n'
             'Nearby WPs share events. Agreement with a VR does not by itself calibrate nominal κ or eliminate background-MC statistical fluctuations.')
    return fig


def finalize_comprehensive_plots__nominal_purity(result):
    points = finalize_comprehensive_plots__points(result)
    audit = finalize_comprehensive_plots__table(result, 'nominal_bcd_audit')
    matrix = audit.pivot(index='wp', columns='region', values='max_S_B').reindex(index=points.wp, columns=list('BCD')).to_numpy(float)
    finite = matrix[finalize_comprehensive_plots_np.isfinite(matrix) & (matrix > 0)]
    vmin = min(.01, finite.min() * .7) if len(finite) else .001
    vmax = max(.2, finite.max() * 1.3) if len(finite) else 1.
    fig, axes = finalize_comprehensive_plots__canvas('Nominal B/C/D contamination: MC safety check', cols=1, height=7.7, bottom=.22)
    ax = axes[0, 0]
    mesh = ax.imshow(finalize_comprehensive_plots_np.ma.masked_invalid(finalize_comprehensive_plots_np.where(matrix > 0, matrix, finalize_comprehensive_plots_np.nan)), aspect='auto',
                     cmap='viridis', norm=finalize_comprehensive_plots_LogNorm(vmin, vmax))
    for i, wp in enumerate(points.wp):
        for j, region in enumerate('BCD'):
            row = audit.loc[audit.wp.eq(wp) & audit.region.eq(region)].iloc[0]
            value = matrix[i, j]
            text = f'{100*value:.2f}%' if finalize_comprehensive_plots_np.isfinite(value) else 'unsupported'
            text += '\n' + ('S/B pass' if bool(row.purity_pass) else 'S/B blocked')
            if bool(row.overlap_original_A):
                text += '\ncontains original blind A'
            elif bool(row.overlap_protected_A):
                text += '\ncontains protected candidate A'
            ax.text(j, i, text, ha='center', va='center', fontsize=9.5, color='white',
                    bbox=dict(facecolor='black', alpha=.55, edgecolor='none', pad=2))
    ax.set(xticks=range(3), xticklabels=list('BCD'), yticks=range(len(points)),
           yticklabels=[label.replace('\n', ' ') for label in finalize_comprehensive_plots__labels(points)])
    ax.grid(False)
    fig.colorbar(mesh, ax=ax, fraction=.025, pad=.025, label='Maximum S/B_MC over 120 hypotheses')
    finalize_comprehensive_plots__caption(fig, 'The 10% test applies separately to B, C and D; nominal A is intentionally excluded from this sideband-access test and remains blind.\n'
             'Passing contamination is not sufficient if a new sideband intersects an already protected A region. All values shown here are MC diagnostics, not nominal DATA.')
    return fig


def finalize_comprehensive_plots__blind_overlap(result):
    points = finalize_comprehensive_plots__points(result).set_index('wp')
    names = [next(name for name in points.index if str(name).lower() in ('reference', 'ref')), 'WP1', 'WP2']
    fig, axes = finalize_comprehensive_plots__canvas('A boundary change can move blinded events into B/C/D', height=7.4, bottom=.22)
    ymax = finalize_comprehensive_plots_np.pi
    for ax, name in zip(axes[0], names):
        tx, ty = float(points.loc[name, 'tx']), float(points.loc[name, 'ty'])
        ax.add_patch(finalize_comprehensive_plots_Rectangle((0, 2.6), .3, ymax - 2.6, facecolor='#d8d0e9', alpha=.8,
                               edgecolor='#6F559C', hatch='///'))
        bounds = {'B': (tx, .3, max(ty, 2.6), ymax),
                  'C': (0, min(tx, .3), 2.6, min(ty, ymax)),
                  'D': (tx, .3, 2.6, min(ty, ymax))}
        for region, (xl, xh, yl, yh) in bounds.items():
            if xh > xl and yh > yl:
                ax.add_patch(finalize_comprehensive_plots_Rectangle((xl, yl), xh - xl, yh - yl,
                                       facecolor=finalize_comprehensive_plots_REGION_COLORS[region], alpha=.5,
                                       edgecolor=finalize_comprehensive_plots_REGION_COLORS[region], hatch='xx'))
                ax.text((xl + xh) / 2, (yl + yh) / 2, region + ' ∩ old A',
                        ha='center', va='center', fontsize=8, rotation=90 if xh-xl < .15 else 0)
        ax.axvline(tx, color='#cc3311', lw=1.5)
        ax.axhline(ty, color='#cc3311', lw=1.5)
        ax.set(xlim=(0, .5), ylim=(2.3, ymax), xlabel='Mu isolation', ylabel='|Δφ| [rad]',
               title=f'{name}: ({tx:g}, {ty:g})')
        ax.text(.02, .03, 'GEOMETRY ONLY · NO DATA VALUES', transform=ax.transAxes, fontsize=8.5)
    handles = [finalize_comprehensive_plots_Patch(facecolor='#d8d0e9', edgecolor='#6F559C', hatch='///', label='Original blind A: iso<0.30, |Δφ|≥2.6'),
               finalize_comprehensive_plots_Line2D([0], [0], color='#cc3311', label='New WP boundaries')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .105), ncol=2, frameon=False)
    finalize_comprehensive_plots__caption(fig, 'WP1 places part of the old A in B. WP2 also places parts of the old A in C and D. This schematic zooms into the relevant angle range.\n'
             'The access audit additionally protects the union of the reference and all five candidate A regions. Removing overlaps and applying unchanged BC/D would no longer describe the original ABCD plane.')
    return fig


def finalize_comprehensive_plots__data_access(result):
    points = finalize_comprehensive_plots__points(result)
    status = finalize_comprehensive_plots__table(result, 'data_sensitivity_status').set_index('wp').loc[points.wp]
    fig, axes = finalize_comprehensive_plots__canvas('Nominal DATA-based expected sensitivity: access status', cols=1, height=7.8, bottom=.20)
    ax = axes[0, 0]
    ax.axis('off')
    rows = []
    reason_labels = {
        'CONTAMINATION_OR_INVALID_MC_SUPPORT': 'Contamination ≥10% or invalid MC support',
        'OVERLAPS_ORIGINAL_BLINDED_A': 'Intersects original blinded A',
        'OVERLAPS_ADDITIONAL_CANDIDATE_A_PROTECTION': 'Intersects additional candidate-A protection',
        'MC_AND_GEOMETRY_CHECKS_PASS_NO_DATA_READ': 'MC and geometry checks pass; DATA not read',
        'NO_NOMINAL_DATA_ACCESS_IN_THIS_NOTEBOOK': 'Nominal DATA not accessed in this notebook',
    }
    for point in points.itertuples():
        row = status.loc[point.wp]
        codes = dict.fromkeys(code.strip() for code in str(row['reason']).replace('|', ';').split(';')
                              if code.strip())
        reason = '\n'.join(reason_labels.get(code, code.replace('_', ' ').capitalize()) for code in codes)
        rows.append([f'{point.wp}\n({point.tx:g}, {point.ty:g})',
                     str(row['status']).replace('_', ' ').title(), reason])
    table = ax.table(cellText=rows, colLabels=['WP', 'Nominal DATA access / sensitivity', 'Reason'],
                     cellLoc='left', colLoc='left', colWidths=[.12, .28, .60], bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('#d6dde5')
        if i == 0:
            cell.set_facecolor('#17263b'); cell.set_text_props(color='white', weight='bold')
        else:
            cell.set_facecolor('#f1f4f8' if i % 2 else '#ffffff')
    finalize_comprehensive_plots__caption(fig, 'Only complete, contamination-safe sidebands outside every protected A region could support a DATA B×C/D estimate. No hole-subtracted substitute is used.\n'
             'No nominal DATA yield or significance is manufactured for blocked cases. With A still blind, a future allowed calculation would be expected sensitivity with DATA-constrained background, not observed significance.')
    return fig


def finalize_comprehensive_plots__mass_rows(result, source, plane):
    points = finalize_comprehensive_plots__points(result)
    rows = result['mass']['tables']['selected_closure']
    rows = rows.loc[rows.source.eq(source) & rows.plane.eq(plane)].set_index('wp').loc[points.wp]
    if source == 'DATA' and not rows.loc[~rows.approved, ['A', 'pred', 'kappa', 'A_error', 'pred_error', 'kappa_error']].isna().all().all():
        raise AssertionError('Blocked mass-inclusive DATA values must remain hidden.')
    return rows


def finalize_comprehensive_plots__intermediate_rows(result, source, plane):
    points = finalize_comprehensive_plots__points(result)
    full = result['focus']['tables']['full_closure']
    frame = full.loc[full.source.eq(source) & full.plane.eq(plane)]
    rows = []
    for point in points.itertuples():
        chosen = finalize_comprehensive_plots__cut_rows(frame, point)
        if len(chosen) != 1:
            raise ValueError('Missing unique intermediate-mass comparison WP.')
        rows.append(chosen.iloc[0])
    return finalize_comprehensive_plots_pd.DataFrame(rows).reset_index(drop=True)


def finalize_comprehensive_plots__mass_closure(result):
    points = finalize_comprehensive_plots__points(result)
    fig, axes = finalize_comprehensive_plots__canvas('Mass-inclusive closure at the same boundaries', rows=3, height=10., bottom=.21, top=.85)
    x = finalize_comprehensive_plots_np.arange(len(points))
    for col, plane in enumerate(finalize_comprehensive_plots_PLANES):
        data = finalize_comprehensive_plots__mass_rows(result, 'DATA', plane)
        mc = finalize_comprehensive_plots__mass_rows(result, 'MC', plane)
        old_data = finalize_comprehensive_plots__intermediate_rows(result, 'DATA', plane)
        old_mc = finalize_comprehensive_plots__intermediate_rows(result, 'MC', plane)
        axes[0, col].errorbar(x-.06, data.A, yerr=data.A_error, fmt='o', color='#222222', capsize=3, label='Inclusive DATA A')
        axes[0, col].errorbar(x+.06, data.pred, yerr=data.pred_error, fmt='s', color='#D55E00', capsize=3, label='Inclusive DATA B×C/D')
        axes[0, col].set(title=plane, ylim=(0, None))
        finalize_comprehensive_plots__closure(axes[1, col], x, data, '#222222', 'Inclusive DATA', shift=-.11)
        finalize_comprehensive_plots__closure(axes[1, col], x, mc, '#E69F00', 'Inclusive MC', marker='s', shift=0)
        finalize_comprehensive_plots__closure(axes[1, col], x, old_data, '#0072B2', '150–300 GeV DATA', marker='D', shift=.11)
        axes[1, col].axhspan(.9, 1.1, color='#83B881', alpha=.17)
        axes[1, col].axhline(1, color='#5F775B', ls='--', lw=1)
        axes[1, col].set_ylim(0, 2)
        finalize_comprehensive_plots__positive_line(axes[2, col], x, mc.max_S_B, color='#222222', marker='o', label='Inclusive mass')
        finalize_comprehensive_plots__positive_line(axes[2, col], x, old_mc.max_S_B, color='#0072B2', marker='D', label='150–300 GeV')
        axes[2, col].axhline(.1, color='#B52B25', ls='--', lw=1)
        axes[2, col].set_yscale('log')
        for ax in axes[:, col]:
            ax.set_xticks(x, finalize_comprehensive_plots__labels(points) if ax is axes[2, col] else [])
        blocked = ~data.approved.to_numpy(bool)
        if blocked.any():
            axes[0, col].text(.02, .96, 'Blocked DATA WPs: ' + ', '.join(points.loc[blocked, 'wp']),
                              transform=axes[0, col].transAxes, va='top', fontsize=8)
    axes[0, 0].set_ylabel('DATA yield')
    axes[1, 0].set_ylabel('κ = A / (B×C/D)')
    axes[2, 0].set_ylabel('Max S/B_MC · 120 models × ABCD')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles2, labels2 = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles+handles2, labels+labels2, loc='lower center', bbox_to_anchor=(.5, .10), ncol=3, fontsize=8.7, frameon=False)
    finalize_comprehensive_plots__caption(fig, 'Inclusive mass means physical MJJ≥0 with no upper bound, including overflow. Only PP/PD/DP and partner isolation <0.5 are used; DD is excluded.\n'
             'Mass-inclusive DATA are independently gated with all 120 hypotheses. The inclusive and 150–300 GeV samples overlap; their closure points are not independent measurements.')
    return fig


def finalize_comprehensive_plots__mass_components(result):
    points = finalize_comprehensive_plots__points(result)
    components = result['mass']['tables']['mass_components']
    fig, axes = finalize_comprehensive_plots__canvas('Does low-mass background dominate the inclusive VR?', rows=2, height=8.7, bottom=.23)
    x = finalize_comprehensive_plots_np.arange(len(points))
    for col, plane in enumerate(finalize_comprehensive_plots_PLANES):
        part = components.loc[components.plane.eq(plane)]
        bottom = finalize_comprehensive_plots_np.zeros(len(points))
        for mass in ('low', 'intermediate', 'high'):
            a = part.loc[part.mass_slice.eq(mass) & part.region.eq('A')].set_index('wp').reindex(points.wp)
            values = a.background_fraction.to_numpy(float)
            axes[0, col].bar(x, values, bottom=bottom, color=finalize_comprehensive_plots_MASS_COLORS[mass], label=finalize_comprehensive_plots_MASS_LABELS[mass], width=.65)
            bottom += finalize_comprehensive_plots_np.nan_to_num(values)
            maxima = part.loc[part.mass_slice.eq(mass)].groupby('wp').max_S_B.agg(
                lambda values: values.max() if len(values) == 4 and finalize_comprehensive_plots_np.isfinite(values).all() else finalize_comprehensive_plots_np.nan
            ).reindex(points.wp)
            finalize_comprehensive_plots__positive_line(axes[1, col], x, maxima, color=finalize_comprehensive_plots_MASS_COLORS[mass], marker='o', label=finalize_comprehensive_plots_MASS_LABELS[mass])
        axes[0, col].set(title=plane, ylim=(0, 1.08))
        axes[1, col].axhline(.1, color='#B52B25', ls='--', lw=1)
        axes[1, col].set_yscale('log')
        for row in range(2):
            axes[row, col].set_xticks(x, finalize_comprehensive_plots__labels(points) if row else [])
    axes[0, 0].set_ylabel('Fraction of inclusive background MC A')
    axes[1, 0].set_ylabel('Slice maximum S/B_MC over models and ABCD')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .12), ncol=3, frameon=False)
    finalize_comprehensive_plots__caption(fig, 'The low/intermediate/high mass intervals are disjoint and sum to the inclusive physical-mass plane. High mass includes overflow; mass underflow is excluded.\n'
             'The top row tests the proposed low-mass background dilution. Slice contamination maxima are diagnostics; DATA access uses the recomputed inclusive S/B, not a sum of maxima. Unsupported slice ratios remain gaps.')
    return fig


def finalize_comprehensive_plots__mass_purity(result):
    points = finalize_comprehensive_plots__points(result)
    purity = result['mass']['tables']['selected_purity']
    fig, axes = finalize_comprehensive_plots__canvas('Mass-inclusive contamination in each ABCD region', height=7.4, bottom=.23)
    x = finalize_comprehensive_plots_np.arange(len(points))
    for ax, plane in zip(axes[0], finalize_comprehensive_plots_PLANES):
        part = purity.loc[purity.plane.eq(plane)]
        for region in 'ABCD':
            values = part.loc[part.region.eq(region)].set_index('wp').reindex(points.wp).max_S_B
            finalize_comprehensive_plots__positive_line(ax, x, values, color=finalize_comprehensive_plots_REGION_COLORS[region], marker='o', ms=4, label=region)
        ax.axhline(.1, color='#B52B25', ls='--', lw=1.2)
        ax.set(yscale='log', title=plane, xticks=x, xticklabels=finalize_comprehensive_plots__labels(points))
    axes[0, 0].set_ylabel('Maximum S/B_MC over 120 hypotheses')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .12), ncol=4, frameon=False)
    finalize_comprehensive_plots__caption(fig, 'Mass-inclusive PP/PD/DP are disjoint from nominal DD. Nevertheless, every ABCD region must pass the positive-MC-support and all-120-hypothesis S/B<0.10 test.\n'
             'The maximizing signal hypothesis can differ between regions and WPs. The detailed table retains its identity and MC statistical uncertainty.')
    return fig


def finalize_comprehensive_plots__mass_distribution(result, wp, source):
    point = finalize_comprehensive_plots__points(result).set_index('wp').loc[wp]
    mass = result['mass']
    mc_planes = [mass['planes'][key]['VR'] for key in finalize_comprehensive_plots_PLANES]
    xx = finalize_comprehensive_plots_reference_plots._opt_distribution_edges(mc_planes, 'xedges', 2.)
    yy = finalize_comprehensive_plots_reference_plots._opt_distribution_edges(mc_planes, 'yedges', finalize_comprehensive_plots_np.pi)
    arrays = []
    for plane in finalize_comprehensive_plots_PLANES:
        selected = mass['tables']['selected_closure']
        rows = selected.loc[selected.wp.eq(wp) & selected.plane.eq(plane) & selected.source.eq(source)]
        if len(rows) != 1:
            raise ValueError('Every distribution needs one selected integrated WP gate.')
        if source == 'DATA' and not bool(rows.iloc[0].approved):
            arrays.append(None)
            continue
        raw = mass['data_planes'][plane] if source == 'DATA' else mass['planes'][plane]['VR']
        if raw is None:
            raise ValueError('An approved DATA distribution requires a loaded plane.')
        folded = finalize_comprehensive_plots_reference_plots._opt_fold_distribution(raw, xx, yy, True)[0]
        if source == 'DATA':
            folded = finalize_comprehensive_plots_reference_plots._opt_mask_distribution(folded, mass['distribution_purity'][plane], xx, yy)
        arrays.append(folded)
    norm, cmap, color_label = finalize_comprehensive_plots_reference_plots._opt_distribution_norm([a for a in arrays if a is not None])
    fig, axes = finalize_comprehensive_plots__canvas(f'Mass-inclusive {source} distributions · {wp} ({point.tx:g}, {point.ty:g})',
                        height=7., bottom=.23, top=.83)
    mesh = None
    for ax, plane, array in zip(axes[0], finalize_comprehensive_plots_PLANES, arrays):
        ax.set(title=plane, xlabel='Mu-LJ isolation', ylabel='|Δφ| [rad]',
               xlim=(0, 2), ylim=(0, finalize_comprehensive_plots_np.pi), facecolor='#eeeeee')
        ax.grid(False)
        if array is None:
            finalize_comprehensive_plots_reference_plots._opt_unavailable(ax, 'DATA blinded\nIntegrated WP gate not passed', gate=True)
            continue
        mesh = ax.pcolormesh(xx, yy, array.T, norm=norm, cmap=cmap, shading='flat')
        if source == 'DATA':
            hidden = ~mass['distribution_purity'][plane]['approved']
            for ix, iy in finalize_comprehensive_plots_np.argwhere(hidden):
                ax.add_patch(finalize_comprehensive_plots_Rectangle((xx[ix], yy[iy]), xx[ix+1]-xx[ix], yy[iy+1]-yy[iy],
                                       facecolor='#e6e6e6', edgecolor='#777777', hatch='///', lw=0., zorder=2))
            ax.text(.02, .98, f'Blinded bins: {hidden.sum()}/{hidden.size}', transform=ax.transAxes,
                    va='top', fontsize=8.5, bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
        ax.axvline(point.tx, color='#cc3311', lw=1.5)
        ax.axhline(point.ty, color='#cc3311', lw=1.5)
        for region, (x, y) in zip('ABCD', ((point.tx/2, (point.ty+finalize_comprehensive_plots_np.pi)/2),
                                          ((point.tx+2)/2, (point.ty+finalize_comprehensive_plots_np.pi)/2),
                                          (point.tx/2, point.ty/2), ((point.tx+2)/2, point.ty/2))):
            ax.text(x, y, region, ha='center', va='center', color='white', fontsize=11,
                    bbox=dict(facecolor='black', alpha=.65, edgecolor='none', pad=2))
    fig.subplots_adjust(right=.88)
    if mesh is not None:
        fig.colorbar(mesh, cax=fig.add_axes([.91, .25, .015, .53]),
                     label=color_label.replace('Weighted events', 'Events') if source == 'DATA' else color_label)
    note = ('Hatching: displayed-bin max S/B≥10% or invalid MC support. Hidden values never set the DATA color scale.'
            if source == 'DATA' else 'MC values are shown without a DATA purity mask.')
    finalize_comprehensive_plots__caption(fig, 'PP/PD/DP only; 0≤partner iso<0.5; physical MJJ≥0 with no upper mass bound. Nominal DD DATA are never projected.\n'
             'Exact display bins reach iso=2; higher-iso tails/overflow fold into the last bin, angle flows into edge bins. Isolation underflow is excluded.\n' + note)
    return fig


def finalize_comprehensive_plots__mass_scan(result, config, plane):
    points = finalize_comprehensive_plots__points(result)
    points['is_reference'] = points.wp.astype(str).str.lower().isin(['reference', 'ref'])
    points['is_top5'] = ~points.is_reference
    points['rank'] = [0 if is_ref else int(str(wp).removeprefix('WP'))
                      for wp, is_ref in zip(points.wp, points.is_reference)]
    full = result['mass']['tables']['grid'].copy()
    full['scope'] = 'VR2_' + full.plane
    adapter = dict(tables=dict(working_points=points, full_closure=full))
    fig = finalize_comprehensive_plots_local_plots._local_scan_figure(adapter, config, 'VR2_' + plane)
    fig._suptitle.set_text(fig._suptitle.get_text().replace('Intermediate MJJ VR', 'Mass-inclusive VR'))
    for text in fig.texts:
        text.set_text(text.get_text().replace('150 ≤ MJJ < 300 GeV', 'physical MJJ≥0; no upper mass bound'))
    if any('150 ≤ MJJ < 300 GeV' in text.get_text() for text in fig.texts):
        raise AssertionError('A mass-inclusive figure retained the intermediate mass label.')
    return fig


def finalize_comprehensive_plots_comprehensive_figure_definitions():
    entries = [
        ('sb_regions', 'S/B separated by ABCD region', 'Track the 120-model regional maxima and their global envelope at fixed iso=0.20.'),
        ('sb_c_decomposition', 'Region-C signal and background retention', 'Separate denominator shrinkage from switching signal hypotheses; show MC support.'),
        ('nominal_retention_composition', 'Nominal MC SR retention and composition', 'Compare fixed WPs using signal retention and background process yields.'),
        ('nominal_mc_closure', 'Nominal MC SR closure and effective counts', 'Compare direct A with MC BC/D without touching nominal DATA.'),
        ('nominal_bcd_purity', 'Nominal sideband contamination', 'Evaluate B/C/D separately across all 120 signal hypotheses.'),
        ('nominal_blind_overlap', 'Old-A and new-sideband overlap', 'Show geometric blind-mask intersections for reference, WP1 and WP2; no DATA values.'),
        ('nominal_data_access', 'Nominal DATA sensitivity eligibility', 'Record blocked cases without manufacturing a DATA background or significance.'),
        ('mass_inclusive_closure', 'Mass-inclusive closure at the same WPs', 'Compare new inclusive-mass MC/DATA and the intermediate-mass validation.'),
        ('mass_components', 'Mass contributions and dilution check', 'Separate low/intermediate/high mass background and contamination diagnostics.'),
        ('mass_purity_selected', 'Mass-inclusive ABCD contamination', 'Apply the original all-120-model criterion to the changed mass selection.'),
        ('mass_data_planes_Reference', 'Mass-inclusive DATA distribution at the reference', 'Inspect the allowed reference WP with the same displayed-bin purity mask used for the candidates.'),
        ('mass_data_planes_WP1', 'Mass-inclusive DATA distribution at WP1', 'Use both the selected integrated gate and the exact displayed-bin purity mask.'),
        ('mass_data_planes_WP2', 'Mass-inclusive DATA distribution at WP2', 'Keep identical boundaries in MC SR and the disjoint inclusive-mass VR.'),
        ('mass_mc_planes_WP2', 'Mass-inclusive MC distribution at WP2', 'Expose the background support without a DATA mask.'),
        ('mass_scan_PP', 'Mass-inclusive PP stability scans', 'Repeat the native-boundary closure scans after removing both mass bounds.'),
        ('mass_scan_PD', 'Mass-inclusive PD stability scans', 'Compare the primary displaced-partner validation at the same boundaries.'),
        ('mass_scan_DP', 'Mass-inclusive DP stability scans', 'Retain the lower-statistics cross-check with explicit gaps and support flags.'),
    ]
    return finalize_comprehensive_plots_pd.DataFrame(entries, columns=['name', 'title', 'purpose'])


def finalize_comprehensive_plots_make_comprehensive_figures(result, config):
    """Render follow-up figures only; the notebook keeps its original eight."""
    with finalize_comprehensive_plots_plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.titlesize': 13, 'axes.labelsize': 11,
                         'figure.max_open_warning': 0}):
        figures = [
            (finalize_comprehensive_plots__sb_regions(result), 'sb_regions'),
            (finalize_comprehensive_plots__sb_decomposition(result), 'sb_c_decomposition'),
            (finalize_comprehensive_plots__nominal_retention(result), 'nominal_retention_composition'),
            (finalize_comprehensive_plots__nominal_closure(result), 'nominal_mc_closure'),
            (finalize_comprehensive_plots__nominal_purity(result), 'nominal_bcd_purity'),
            (finalize_comprehensive_plots__blind_overlap(result), 'nominal_blind_overlap'),
            (finalize_comprehensive_plots__data_access(result), 'nominal_data_access'),
            (finalize_comprehensive_plots__mass_closure(result), 'mass_inclusive_closure'),
            (finalize_comprehensive_plots__mass_components(result), 'mass_components'),
            (finalize_comprehensive_plots__mass_purity(result), 'mass_purity_selected'),
            (finalize_comprehensive_plots__mass_distribution(result, 'Reference', 'DATA'), 'mass_data_planes_Reference'),
            (finalize_comprehensive_plots__mass_distribution(result, 'WP1', 'DATA'), 'mass_data_planes_WP1'),
            (finalize_comprehensive_plots__mass_distribution(result, 'WP2', 'DATA'), 'mass_data_planes_WP2'),
            (finalize_comprehensive_plots__mass_distribution(result, 'WP2', 'MC'), 'mass_mc_planes_WP2'),
        ]
        figures.extend((finalize_comprehensive_plots__mass_scan(result, config, plane), 'mass_scan_' + plane) for plane in finalize_comprehensive_plots_PLANES)
    if [name for _, name in figures] != finalize_comprehensive_plots_comprehensive_figure_definitions().name.tolist():
        raise AssertionError('Comprehensive figure order differs from the declared definitions.')
    for figure, _ in figures:
        figure.canvas.draw()
        finalize_comprehensive_plots_plt.close(figure)
    return figures


# ========================================================================
# abcd_final_check_plots: original study API
# ========================================================================
"""Focused final-check figures for the fixed 2mu2e WP1/WP2 comparison.

Only validated tables and an MC plane are accepted. No DATA is opened here;
cached VR measurements are displayed only after their own integrated gate.
"""
import textwrap as finalize_final_check_plots_textwrap

import matplotlib.pyplot as finalize_final_check_plots_plt
from matplotlib.lines import Line2D as finalize_final_check_plots_Line2D
from matplotlib.patches import Patch as finalize_final_check_plots_Patch
import numpy as finalize_final_check_plots_np
import pandas as finalize_final_check_plots_pd

finalize_final_check_plots_reference_plots = _FINALIZE_NAMESPACES['abcd_finalize_plots']
finalize_final_check_plots_COMPONENT_COLORS = _FINALIZE_NAMESPACES['abcd_local_scan_plots'].COMPONENT_COLORS


finalize_final_check_plots_WPS = ('Reference', 'WP1', 'WP2')
finalize_final_check_plots_WP_COLORS = {'Reference': '#777777', 'WP1': '#0072B2', 'WP2': '#D55E00'}
finalize_final_check_plots_PLANES = ('PP', 'PD', 'DP')
finalize_final_check_plots_MASSES = ('low', 'intermediate', 'high', 'all')
finalize_final_check_plots_MASS_LABELS = ('0–150', '150–300', '≥300', 'All mass')
finalize_final_check_plots_MASS_COLORS = {'low': '#0072B2', 'intermediate': '#E69F00', 'high': '#CC79A7'}
finalize_final_check_plots_STEPS = ('reference_A', 'iso_removed', 'angle_removed', 'kept_WP2')
finalize_final_check_plots_STEP_LABELS = ('Reference A', 'Removed by iso\n0.20–0.30', 'Removed by |Δφ|\n2.6–2.7', 'Kept in WP2 A')


def finalize_final_check_plots__table(result, name):
    return result['tables'][name].copy()


def finalize_final_check_plots__canvas(title, rows=1, cols=3, height=8., bottom=.22, top=.85):
    fig, axes = finalize_final_check_plots_plt.subplots(rows, cols, figsize=(16, height), squeeze=False)
    fig.suptitle('2μ2e · ' + title, x=.065, y=.97, ha='left', fontsize=19, fontweight='bold')
    fig.subplots_adjust(left=.075, right=.97, top=top, bottom=bottom, wspace=.30, hspace=.35)
    for ax in axes.ravel():
        ax.grid(alpha=.20)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)
    return fig, axes


def finalize_final_check_plots__caption(fig, text):
    wrapped = '\n'.join(finalize_final_check_plots_textwrap.fill(line, width=170) for line in text.splitlines())
    fig.text(.5, .018, wrapped, ha='center', va='bottom', fontsize=9, color='#444444')


def finalize_final_check_plots__wp_labels():
    return ['Reference\n(0.30, 2.6)', 'WP1\n(0.20, 2.6)', 'WP2\n(0.20, 2.7)']


def finalize_final_check_plots__format(value, spec='.3f'):
    return format(float(value), spec) if finalize_final_check_plots_np.isfinite(value) else 'undefined'


def finalize_final_check_plots__style_table(table, size=11):
    table.auto_set_font_size(False)
    table.set_fontsize(size)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#d6dde5')
        if row == 0:
            cell.set_facecolor('#17263b')
            cell.set_text_props(color='white', weight='bold')
        else:
            cell.set_facecolor('#f1f4f8' if row % 2 else '#ffffff')


def finalize_final_check_plots__summary(result):
    nominal = finalize_final_check_plots__table(result, 'nominal_summary').set_index('wp')
    significance = finalize_final_check_plots__table(result, 'significance_summary').set_index('wp')
    audit = finalize_final_check_plots__table(result, 'nominal_bcd_audit')
    data = finalize_final_check_plots__table(result, 'data_sensitivity_status').set_index('wp')
    mass = finalize_final_check_plots__table(result, 'mass_closure')
    metrics = ['Median signal retention / reference', 'Background A / reference',
               'Median Z / reference: known background', 'Median Z / reference: finite MC diagnostic',
               'Effective background MC count in A', 'Nominal MC κ = A / (B×C/D)',
               'Maximum nominal C contamination', 'Nominal DATA estimate',
               'All-mass VR DATA shown (PP/PD/DP)']
    values = []
    for wp in finalize_final_check_plots_WPS:
        row, sig = nominal.loc[wp], significance.loc[wp]
        purity = audit.loc[audit.wp.eq(wp) & audit.region.eq('C'), 'max_S_B'].iloc[0]
        allowed = mass.loc[mass.wp.eq(wp) & mass.mass_slice.eq('all') & mass.source.eq('DATA'), 'display_allowed']
        values.append([f'{sig.signal_retention_median:.1%}', f'{sig.background_retention:.1%}',
                       finalize_final_check_plots__format(sig.z_known_b_ratio_median), finalize_final_check_plots__format(sig.z_mc_stat_diagnostic_ratio_median),
                       finalize_final_check_plots__format(row.neff_A, '.2f'), f'{row.kappa:.3f} ± {row.kappa_error:.3f}',
                       f'{purity:.1%}', str(data.loc[wp, 'status']).replace('_', ' ').title(),
                       f'{int(allowed.sum())} / {len(allowed)}'])
    rows = [[name] + [values[j][i] for j in range(3)] for i, name in enumerate(metrics)]
    fig, axes = finalize_final_check_plots__canvas('WP1 / WP2 final-check scorecard', cols=1, height=8.6, bottom=.22)
    ax = axes[0, 0]
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=['Quantity'] + finalize_final_check_plots__wp_labels(), cellLoc='center',
                     colWidths=[.43, .19, .19, .19], bbox=[0, 0, 1, 1])
    finalize_final_check_plots__style_table(table, 10.7)
    for i in range(1, len(rows)+1):
        table[i, 0].set_text_props(ha='left')
    finalize_final_check_plots__caption(fig, 'Fixed candidates; no new ranking from VR DATA. Nominal DD MC: MJJ≥150 GeV and partner isolation <0.5.\n'
             'Signal summaries describe 60 matching 2Mu2E hypotheses separately. Finite-MC Z is an effective on/off diagnostic, not a full ABCD likelihood or observed significance.\n'
             'A passing purity gate alone does not grant display permission: inclusive DATA are also withheld where component subtraction could reconstruct blinded high-mass yields.\n'
             'The final boundary remains provisional while nominal MC support, contamination and mass-dependent closure are unresolved.')
    return fig


def finalize_final_check_plots__nominal_support(result):
    regions, support = finalize_final_check_plots__table(result, 'nominal_regions'), finalize_final_check_plots__table(result, 'nominal_support')
    fig, axes = finalize_final_check_plots__canvas('Nominal ABCD: yield, effective count and variance support', height=7.6)
    x = finalize_final_check_plots_np.arange(4)
    for index, wp in enumerate(finalize_final_check_plots_WPS):
        rows = regions.loc[regions.wp.eq(wp)].set_index('region').reindex(list('ABCD'))
        weights = support.loc[support.wp.eq(wp)].set_index('region').reindex(list('ABCD'))
        xx = x + (index-1)*.17
        color = finalize_final_check_plots_WP_COLORS[wp]
        axes[0, 0].errorbar(xx, rows['yield'], yerr=rows.error, fmt='o', color=color, capsize=3, label=wp)
        axes[0, 1].plot(xx, rows.neff, 'o', color=color, label=wp)
        axes[0, 2].plot(xx, weights.max_sample_variance_fraction, 'o', color=color, label=wp+' · sample')
        axes[0, 2].plot(xx, weights.max_cell_variance_fraction, 'x', color=color, label=wp+' · native cell')
    axes[0, 0].set(yscale='symlog', ylabel='Weighted yield ± √sumw²')
    axes[0, 1].set(yscale='log', ylabel='N_eff = (sumw)² / sumw²')
    axes[0, 1].axhline(10, color='#888888', ls=':', lw=1)
    axes[0, 2].set(ylabel='Largest share of regional sumw²', ylim=(0, 1.08))
    for ax in axes[0]:
        ax.set_xticks(x, list('ABCD'))
    axes[0, 0].legend(fontsize=9)
    axes[0, 2].legend(fontsize=8, ncol=2, loc='best')
    finalize_final_check_plots__caption(fig, 'Nominal MC only. N_eff and the variance fractions measure weighted histogram support; they are not raw event counts.\n'
             'Sample and native-cell maxima are evaluated separately. A large variance share identifies sensitivity to a small part of the MC support, without identifying individual event weights.')
    return fig


def finalize_final_check_plots__sample_leverage(result):
    frame = finalize_final_check_plots__table(result, 'nominal_samples').query('region == "A"')
    ranking = frame.groupby('sample').variance.sum().sort_values(ascending=False)
    names = list(ranking.head(6).index)
    labels = names + ['Other samples']
    fig, axes = finalize_final_check_plots__canvas('Which MC samples control nominal A?', rows=2, height=9.1, bottom=.19)
    for col, wp in enumerate(finalize_final_check_plots_WPS):
        rows = frame.loc[frame.wp.eq(wp)].set_index('sample')
        for row_index, field in enumerate(('yield', 'variance')):
            total = rows[field].sum()
            values = [rows[field].get(name, 0.) for name in names]
            values.append(rows.loc[~rows.index.isin(names), field].sum())
            fractions = finalize_final_check_plots_np.asarray(values) / total if total > 0 else finalize_final_check_plots_np.full(len(values), finalize_final_check_plots_np.nan)
            colors = [finalize_final_check_plots_COMPONENT_COLORS.get(rows.loc[name, 'component'], '#999999') for name in names] + ['#bbbbbb']
            ax = axes[row_index, col]
            ax.barh(finalize_final_check_plots_np.arange(len(labels)), fractions, color=colors)
            ax.set(yticks=finalize_final_check_plots_np.arange(len(labels)), yticklabels=labels if col == 0 else [], xlim=(0, 1.05))
            ax.invert_yaxis()
            ax.tick_params(axis='y', labelsize=8)
            for i, fraction in enumerate(fractions):
                if finalize_final_check_plots_np.isfinite(fraction):
                    ax.text(min(fraction+.015, .95), i, f'{fraction:.1%}', va='center', fontsize=8)
            ax.set_xlabel('Fraction of MC A yield' if row_index == 0 else 'Fraction of MC A sumw²')
            if row_index == 0:
                ax.set_title(wp)
    fig.subplots_adjust(left=.18, wspace=.12)
    handles = [finalize_final_check_plots_Patch(facecolor=color, label=name) for name, color in finalize_final_check_plots_COMPONENT_COLORS.items()]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.56, .095), ncol=len(handles), frameon=False)
    finalize_final_check_plots__caption(fig, 'The six samples with the largest summed A variance across the three fixed WPs are shown separately; all remaining samples form the final row.\n'
             'These are the same physical background samples at each WP. Fractions and support changes are diagnostics; no sample is removed from the prediction.')
    return fig


def finalize_final_check_plots__migration(result):
    sample = finalize_final_check_plots__table(result, 'migration_samples')
    total = finalize_final_check_plots__table(result, 'migration_summary').set_index('step').reindex(finalize_final_check_plots_STEPS)
    signal = finalize_final_check_plots__table(result, 'signal_migration_summary').set_index('step').reindex(finalize_final_check_plots_STEPS)
    fig, axes = finalize_final_check_plots__canvas('What disappears between reference, WP1 and WP2?', cols=2, height=7.8, bottom=.25)
    x, bottom = finalize_final_check_plots_np.arange(4), finalize_final_check_plots_np.zeros(4)
    for component, color in finalize_final_check_plots_COMPONENT_COLORS.items():
        values = sample.loc[sample.component.eq(component)].groupby('step')['yield'].sum().reindex(finalize_final_check_plots_STEPS, fill_value=0)
        axes[0, 0].bar(x, values, bottom=bottom, color=color, label=component, width=.62)
        bottom += values.to_numpy(float)
    axes[0, 0].errorbar(x, total['yield'], yerr=total.error, fmt='none', color='#222222', capsize=4)
    axes[0, 0].set_ylabel('Background MC yield ± √sumw²')
    mid, low, high = (signal[field].to_numpy(float) for field in ('fraction_median', 'fraction_q10', 'fraction_q90'))
    axes[0, 1].errorbar(x-.06, mid, yerr=[mid-low, high-mid], fmt='o', color='#0072B2', capsize=4,
                      label='Signal: median and 10–90% models')
    axes[0, 1].plot(x+.06, total.reference_fraction, 's--', color='#E69F00', label='Background MC central ratio')
    axes[0, 1].set(ylabel='Yield / own reference A yield', ylim=(0, 1.12))
    for ax in axes[0]:
        ax.set_xticks(x, finalize_final_check_plots_STEP_LABELS)
        ax.legend(fontsize=8.5)
    finalize_final_check_plots__caption(fig, 'Reference A = iso-removed + angle-removed + WP2 A, with disjoint sumw and sumw² checks. WP1 A = angle-removed + WP2 A.\n'
             'Iso-removed: 0.20≤iso<0.30, |Δφ|≥2.6. Angle-removed: iso<0.20, 2.6≤|Δφ|<2.7. Nominal DD and MJJ≥150 GeV are fixed.\n'
             'Signal quantiles span 60 alternative hypotheses, not a confidence interval. Reference-normalized central ratios share events; no independent-ratio error is assigned.')
    return fig


def finalize_final_check_plots__nominal_distribution(result):
    plane = result['planes']['nominal']
    xx = finalize_final_check_plots_reference_plots._opt_distribution_edges([plane], 'xedges', 2.)
    yy = finalize_final_check_plots_reference_plots._opt_distribution_edges([plane], 'yedges', finalize_final_check_plots_np.pi)
    values = finalize_final_check_plots_reference_plots._opt_fold_distribution(plane, xx, yy, True)[0]
    var_plane = dict(plane, values=plane['variances'])
    variance = finalize_final_check_plots_reference_plots._opt_fold_distribution(var_plane, xx, yy, True)[0]
    neff = finalize_final_check_plots_np.divide(values*values, variance, out=finalize_final_check_plots_np.full_like(values, finalize_final_check_plots_np.nan), where=variance > 0)
    fig, axes = finalize_final_check_plots__canvas('Nominal MC near the candidate boundaries', cols=2, height=7.7, bottom=.24)
    for ax, array, title in zip(axes[0], (values, neff), ('Background MC yield', 'Effective MC count per native display bin')):
        norm, cmap, _ = finalize_final_check_plots_reference_plots._opt_distribution_norm([array[finalize_final_check_plots_np.isfinite(array)]])
        mesh = ax.pcolormesh(xx, yy, array.T, cmap=cmap, norm=norm, shading='flat')
        ax.set(xlim=(0, .5), ylim=(0, finalize_final_check_plots_np.pi), xlabel='Mu-LJ isolation', ylabel='|Δφ| [rad]', title=title)
        ax.grid(False)
        ax.axvspan(.2, .3, ymin=2.6/finalize_final_check_plots_np.pi, ymax=1, color='#E69F00', alpha=.15)
        ax.axvline(.3, color=finalize_final_check_plots_WP_COLORS['Reference'], lw=1.7, ls=':')
        ax.axvline(.2, color=finalize_final_check_plots_WP_COLORS['WP1'], lw=1.7)
        ax.axhline(2.6, color=finalize_final_check_plots_WP_COLORS['WP1'], lw=1.7)
        ax.axhline(2.7, color=finalize_final_check_plots_WP_COLORS['WP2'], lw=1.7, ls='--')
        fig.colorbar(mesh, ax=ax, fraction=.045, pad=.025, label='Weighted events' if array is values else 'N_eff')
    handles = [finalize_final_check_plots_Line2D([0], [0], color=finalize_final_check_plots_WP_COLORS['Reference'], ls=':', label='Reference iso=0.30; |Δφ|=2.6'),
               finalize_final_check_plots_Line2D([0], [0], color=finalize_final_check_plots_WP_COLORS['WP1'], label='WP1 iso=0.20; |Δφ|=2.6'),
               finalize_final_check_plots_Line2D([0], [0], color=finalize_final_check_plots_WP_COLORS['WP2'], ls='--', label='WP2 iso=0.20; |Δφ|=2.7')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .12), ncol=3, frameon=False, fontsize=9)
    finalize_final_check_plots__caption(fig, 'MC only; no nominal DATA display. The view is zoomed to iso≤0.5; all ABCD calculations retain the allowed higher-isolation bins and overflow.\n'
             'Angle flows are folded into edge bins, isolation underflow is excluded. N_eff is calculated after exact display folding and is not the raw event multiplicity.')
    return fig


def finalize_final_check_plots__mass_closure(result):
    frame = finalize_final_check_plots__table(result, 'mass_closure')
    fig, axes = finalize_final_check_plots__canvas('Mass-dependent closure at fixed WP1 / WP2 boundaries', rows=3, height=11.0, bottom=.17, top=.87)
    x = finalize_final_check_plots_np.arange(4)
    for i, wp in enumerate(finalize_final_check_plots_WPS):
        for j, plane in enumerate(finalize_final_check_plots_PLANES):
            ax = axes[i, j]
            selected = frame.loc[frame.wp.eq(wp) & frame.plane.eq(plane)]
            mc = selected.loc[selected.source.eq('MC')].set_index('mass_slice').reindex(finalize_final_check_plots_MASSES)
            data = selected.loc[selected.source.eq('DATA')].set_index('mass_slice').reindex(finalize_final_check_plots_MASSES)
            if not data.loc[~data.display_allowed, ['A', 'B', 'C', 'D', 'pred', 'kappa']].isna().all().all():
                raise AssertionError('A protected mass-slice DATA value is visible.')
            finalize_final_check_plots_reference_plots._opt_reference_errorbar(ax, x-.07, mc.kappa, error=mc.kappa_error,
                                                    fmt='s', color='#E69F00', capsize=3, label='Background MC')
            finalize_final_check_plots_reference_plots._opt_reference_errorbar(ax, x+.07, data.kappa, error=data.kappa_error,
                                                    fmt='o', color='#222222', capsize=3, label='Allowed cached DATA')
            # Include the measured low-mass DATA points near kappa=2.2, while
            # explicitly annotating the much larger sparse high-mass MC values.
            ax.set_ylim(0, 3)
            for source, selected_rows, offset, note_y, color in (
                    ('MC', mc, -.07, .96, '#8a4b00'), ('DATA', data, .07, .84, '#333333')):
                values = selected_rows.kappa.to_numpy(float)
                errors = selected_rows.kappa_error.to_numpy(float)
                finite = finalize_final_check_plots_np.isfinite(values)
                outside = finite & ((values < 0) | (values > 3))
                clipped = finite & finalize_final_check_plots_np.isfinite(errors) & ((values-errors < 0) | (values+errors > 3))
                if clipped.any():
                    ax.text(.025, note_y, f'{source}: {int(clipped.sum())} error ranges clipped',
                            transform=ax.transAxes, va='top', fontsize=7, color=color,
                            bbox=dict(facecolor='white', alpha=.8, edgecolor='none', pad=1))
                for index in finalize_final_check_plots_np.flatnonzero(outside):
                    location = 2.92 if values[index] > 3 else .08
                    ax.scatter(x[index]+offset, location, marker='^' if values[index] > 3 else 'v',
                               color=color, s=26, zorder=6)
                    ax.text(x[index]+offset, 2.60 if values[index] > 3 else .35,
                            f'{source} κ={values[index]:.1f}', ha='center', va='center', fontsize=7.5,
                            color=color, bbox=dict(facecolor='white', alpha=.85, edgecolor='none', pad=1))
            ax.axhline(1, color='#5F775B', ls='--', lw=1)
            ax.axhspan(.9, 1.1, color='#83B881', alpha=.17)
            blocked = ~data.display_allowed.to_numpy(bool)
            ax.scatter(x[blocked]+.07, finalize_final_check_plots_np.full(blocked.sum(), -.07), transform=ax.get_xaxis_transform(),
                       marker='x', color='#777777', s=25, clip_on=False)
            ax.set(xticks=x, xticklabels=finalize_final_check_plots_MASS_LABELS if i == 2 else [])
            if i == 0:
                ax.set_title(plane + (' · auxiliary' if plane == 'DP' else ' · primary'))
            if j == 0:
                ax.set_ylabel(wp + '\nκ = A / (B×C/D)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(finalize_final_check_plots_Line2D([0], [0], color='#777777', marker='x', ls='none'))
    labels.append('DATA blocked / not evaluated')
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .085), ncol=3, frameon=False, fontsize=9)
    finalize_final_check_plots__caption(fig, 'Mass intervals are in GeV. The high-mass interval includes overflow; all mass is their physical union. Each DATA selection has its own 120-hypothesis × ABCD gate.\n'
             'Reference inclusive DATA are additionally withheld to prevent reconstructing blinded high-mass yields by subtraction. Only display-allowed cached DATA are reused.\n'
             'Values outside the 0–3 frame are marked and printed; clipped error ranges are counted. All-mass and its components share MC events.\n'
             'MC error bars use the established sumw² propagation; unsupported ratios remain gaps.')
    return fig


def finalize_final_check_plots__mass_composition(result):
    frame = finalize_final_check_plots__table(result, 'mass_composition').query('region == "A"')
    fig, axes = finalize_final_check_plots__canvas('Mass and process composition of the validation A region', rows=2, height=9.1, bottom=.20)
    x = finalize_final_check_plots_np.arange(3)
    process_index = finalize_final_check_plots_pd.MultiIndex.from_product([('WP1', 'WP2'), finalize_final_check_plots_MASSES[:3]], names=['wp', 'mass_slice'])
    process_x = finalize_final_check_plots_np.arange(len(process_index))
    process_labels = [f'{wp}\n{finalize_final_check_plots_MASS_LABELS[finalize_final_check_plots_MASSES.index(mass)]}' for wp, mass in process_index]
    for j, plane in enumerate(finalize_final_check_plots_PLANES):
        part = frame.loc[frame.plane.eq(plane)]
        totals = part.loc[part.mass_slice.eq('all')].groupby('wp')['yield'].sum().reindex(finalize_final_check_plots_WPS)
        bottom = finalize_final_check_plots_np.zeros(3)
        for mass, color in finalize_final_check_plots_MASS_COLORS.items():
            counts = part.loc[part.mass_slice.eq(mass)].groupby('wp')['yield'].sum().reindex(finalize_final_check_plots_WPS)
            values = (counts / totals.replace(0, finalize_final_check_plots_np.nan)).to_numpy(float)
            axes[0, j].bar(x, values, bottom=bottom, color=color, label=finalize_final_check_plots_MASS_LABELS[finalize_final_check_plots_MASSES.index(mass)], width=.62)
            bottom += finalize_final_check_plots_np.nan_to_num(values)
        process_totals = part.groupby(['wp', 'mass_slice'])['yield'].sum().reindex(process_index)
        bottom = finalize_final_check_plots_np.zeros(len(process_index))
        for component, color in finalize_final_check_plots_COMPONENT_COLORS.items():
            counts = part.loc[part.component.eq(component)].groupby(['wp', 'mass_slice'])['yield'].sum().reindex(process_index, fill_value=0)
            values = (counts / process_totals.where(process_totals > 0)).to_numpy(float)
            axes[1, j].bar(process_x, values, bottom=bottom, color=color, label=component, width=.67)
            bottom += finalize_final_check_plots_np.nan_to_num(values)
        unsupported = ~finalize_final_check_plots_np.isfinite(process_totals) | process_totals.le(0)
        for position in process_x[unsupported.to_numpy(bool)]:
            axes[1, j].text(position, .50, 'No MC A support', rotation=90, ha='center', va='center',
                            fontsize=8, color='#777777')
        axes[1, j].axvline(2.5, color='#999999', lw=.8, ls=':')
        axes[0, j].set_title(plane)
        axes[0, j].set(ylim=(0, 1.08), xticks=x, xticklabels=finalize_final_check_plots__wp_labels())
        axes[1, j].set(ylim=(0, 1.08), xticks=process_x, xticklabels=process_labels)
        axes[1, j].tick_params(axis='x', labelsize=8.5)
    axes[0, 0].set_ylabel('Fraction of all-mass MC A: mass interval')
    axes[1, 0].set_ylabel('Fraction of each mass-slice MC A: process')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles2, labels2 = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles+handles2, labels+labels2, loc='lower center', bbox_to_anchor=(.5, .085), ncol=7, frameon=False, fontsize=9)
    finalize_final_check_plots__caption(fig, 'MC background composition only. The upper row checks low-mass dilution; the lower row compares the process mixture within each separate mass slice at WP1 and WP2.\n'
             'Mass labels are in GeV: [0,150), [150,300), [300,∞). Unsupported A fractions remain blank and are labelled. Signal hypotheses are never added together.')
    return fig


def finalize_final_check_plots__mass_mixing(result):
    frame = finalize_final_check_plots__table(result, 'mass_mixing')
    fig, axes = finalize_final_check_plots__canvas('Pooling masses changes the ABCD prediction', height=7.5)
    x = finalize_final_check_plots_np.arange(3)
    for ax, plane in zip(axes[0], finalize_final_check_plots_PLANES):
        rows = frame.loc[frame.plane.eq(plane)].set_index('wp').reindex(finalize_final_check_plots_WPS)
        ax.plot(x-.05, rows.pooled_prediction, 'o', color='#222222', label='(ΣB) × (ΣC) / (ΣD)')
        ax.plot(x+.05, rows.sum_slice_predictions, 's', color='#0072B2', label='Σ (B_i × C_i / D_i)')
        ax.set(title=plane, xticks=x, xticklabels=finalize_final_check_plots__wp_labels(), yscale='symlog')
        for i, row in enumerate(rows.itertuples()):
            label = (f'{row.Index}: Δ / pooled = {row.mixing_fraction:.1%}' if finalize_final_check_plots_np.isfinite(row.mixing_fraction)
                     else f'{row.Index}: slice prediction unsupported')
            ax.text(.98, .96-i*.065, label, transform=ax.transAxes, va='top', ha='right', fontsize=8.5,
                    bbox=dict(facecolor='white', alpha=.85, edgecolor='none', pad=1))
    axes[0, 0].set_ylabel('Predicted weighted MC yield in A')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .12), ncol=2, frameon=False)
    finalize_final_check_plots__caption(fig, 'Δ is pooled prediction minus the sum of disjoint-slice predictions. Each slice term requires a supported positive D; otherwise the comparison is left undefined.\n'
             'These central-value diagnostics use overlapping MC events and have no independent-difference error bars. A pooling effect is not a nominal-DD correction or a proof that each slice closes.')
    return fig


def finalize_final_check_plots__c_contamination(result):
    audit = finalize_final_check_plots__table(result, 'nominal_bcd_audit')
    top = finalize_final_check_plots__table(result, 'c_contamination_top')
    fig, axes = finalize_final_check_plots__canvas('Nominal C contamination and the models that drive it', cols=2, height=8.6, bottom=.22)
    x = finalize_final_check_plots_np.arange(3)
    for i, wp in enumerate(finalize_final_check_plots_WPS):
        rows = audit.loc[audit.wp.eq(wp)].set_index('region').reindex(list('BCD'))
        axes[0, 0].plot(x+(i-1)*.10, rows.max_S_B, 'o', color=finalize_final_check_plots_WP_COLORS[wp], label=wp)
    axes[0, 0].axhline(.1, color='#B52B25', ls='--', label='10% criterion')
    axes[0, 0].set(yscale='log', xticks=x, xticklabels=list('BCD'), ylabel='Maximum S/B_MC over 120 hypotheses')
    axes[0, 0].legend(fontsize=9)
    rows = []
    for wp in ('WP1', 'WP2'):
        for entry in top.loc[top.wp.eq(wp)].sort_values('rank').head(3).itertuples():
            rows.append([wp, str(entry.sample), f'{entry.S_B_C:.1%}'])
    axes[0, 1].axis('off')
    table = axes[0, 1].table(cellText=rows, colLabels=['WP', 'Largest C-contamination hypotheses', 'S/B in C'],
                            cellLoc='left', colWidths=[.13, .70, .17], bbox=[-.02, .12, 1.06, .76])
    finalize_final_check_plots__style_table(table, 9)
    finalize_final_check_plots__caption(fig, 'B/C/D are checked separately with all 120 hypotheses. Low contamination in B or D does not grant access when a shifted sideband intersects protected A.\n'
             'The C model identities are retained explicitly. The same model must be used when comparing WP1 with WP2 or evaluating signal leakage in the added angle band.')
    return fig


def finalize_final_check_plots__significance(result):
    rows = finalize_final_check_plots__table(result, 'significance_summary').set_index('wp').reindex(finalize_final_check_plots_WPS)
    fig, axes = finalize_final_check_plots__canvas('Expected MC sensitivity with finite background statistics', cols=2, height=7.7, bottom=.24)
    x = finalize_final_check_plots_np.arange(3)
    metrics = [('z_known_b', '#0072B2', 'Known background'),
               ('z_mc_stat_diagnostic', '#D55E00', 'Finite-MC diagnostic')]
    for i, (metric, color, label) in enumerate(metrics):
        for ax, suffix in zip(axes[0], ('', '_ratio')):
            mid = rows[metric+suffix+'_median'].to_numpy(float)
            low = rows[metric+suffix+'_q10'].to_numpy(float)
            high = rows[metric+suffix+'_q90'].to_numpy(float)
            ax.errorbar(x+(i-.5)*.14, mid, yerr=[mid-low, high-mid], fmt='o', color=color, capsize=4, label=label)
    axes[0, 0].set_ylabel('Expected counting Z: median and 10–90% models')
    axes[0, 1].set_ylabel('Own-model Z / own-model reference Z')
    axes[0, 1].axhline(1, color='#777777', ls=':')
    for ax in axes[0]:
        ax.set_xticks(x, finalize_final_check_plots__wp_labels())
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9)
    finalize_final_check_plots__caption(fig, 'The existing effective on/off Asimov diagnostic uses background variance v=sumw² in nominal A. It is compared with the known-background limit for each signal model.\n'
             'Relative gains use the reference penalized by the same metric; a larger ratio does not mean a larger absolute Z. Model ranges are not confidence intervals.\n'
             'No validated closure nuisance, signal-leakage likelihood or nominal DATA enters this diagnostic.')
    return fig


def finalize_final_check_plots__paired(result):
    frame = finalize_final_check_plots__table(result, 'paired_comparison')
    frame = frame.loc[frame.matching_decay].sort_values('sample')
    if len(frame) != 60 or frame['sample'].duplicated().any():
        raise AssertionError('The paired comparison requires 60 separate matching hypotheses.')
    fig, axes = finalize_final_check_plots__canvas('WP2 versus WP1: compare the same signal hypotheses', height=7.5)
    metrics = [('z_known_b_ratio_WP2_WP1', 'Known-background Z₂ / Z₁'),
               ('z_mc_stat_diagnostic_ratio_WP2_WP1', 'Finite-MC diagnostic Z₂ / Z₁'),
               ('signal_retention_WP2_WP1', 'Signal A₂ / A₁')]
    x = finalize_final_check_plots_np.arange(1, len(frame)+1)
    for ax, (field, title) in zip(axes[0], metrics):
        values = frame[field].to_numpy(float)
        ax.scatter(x, values, s=18, color='#0072B2')
        ax.axhline(1, color='#777777', ls=':')
        finite = finalize_final_check_plots_np.isfinite(values)
        if finite.any():
            ax.axhline(finalize_final_check_plots_np.median(values[finite]), color='#D55E00', lw=1, label='Median')
        ax.set(title=title, xlabel='Signal hypothesis index (same ordering)', ylabel='WP2 / WP1')
        ax.text(.03, .97, f'>1: {int(finalize_final_check_plots_np.sum(values[finite] > 1))}/{int(finite.sum())}',
                transform=ax.transAxes, va='top', fontsize=9)
    finalize_final_check_plots__caption(fig, 'Each point compares one identical signal hypothesis at both WPs. All panels use the same alphabetical model ordering; model IDs and support flags are in paired_comparison.csv.\n'
             'These are central ratios with shared events, not independent statistical measurements. No new WP is selected using VR DATA.')
    return fig


def finalize_final_check_plots__safety(result):
    audit = finalize_final_check_plots__table(result, 'nominal_bcd_audit')
    fig, axes = finalize_final_check_plots__canvas('Final DATA-access checks remain separate from MC ranking', cols=1, height=8.4, bottom=.23)
    rows = []
    for wp in finalize_final_check_plots_WPS:
        for region in 'BCD':
            row = audit.loc[audit.wp.eq(wp) & audit.region.eq(region)].iloc[0]
            protection = 'Original A overlap' if row.overlap_original_A else (
                'Additional candidate-A overlap' if row.overlap_protected_A else 'No protected-A overlap')
            rows.append([wp, region, f'{row.max_S_B:.2%}', 'Pass' if row.purity_pass else 'Fail', protection,
                         'Not read' if not row.nominal_data_evaluated else 'ERROR'])
    ax = axes[0, 0]
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=['WP', 'Sideband', 'Max S/B_MC', 'Purity', 'Geometric protection', 'Nominal DATA'],
                     cellLoc='center', colWidths=[.12, .10, .15, .10, .36, .17], bbox=[0, 0, 1, 1])
    finalize_final_check_plots__style_table(table, 10.3)
    finalize_final_check_plots__caption(fig, 'The original blind A and the previously protected union of all five candidate A regions remain protected. Narrowing this notebook to WP1/WP2 does not relax that policy.\n'
             'A complete B/C/D estimate requires both contamination and geometry checks. No protected pieces are removed and substituted into the unchanged BC/D formula.\n'
             'Nominal DATA A stays blind. This final-check notebook records unresolved conditions; it does not authorize unblinding or finalize a boundary.')
    return fig


def finalize_final_check_plots_final_check_figure_definitions():
    entries = [
        ('final_summary', 'WP1 / WP2 final-check scorecard', 'Collect the fixed-candidate MC, contamination and access diagnostics.'),
        ('nominal_abcd_support', 'Nominal ABCD support', 'Compare yields, effective counts and the concentration of MC variance.'),
        ('nominal_sample_leverage', 'Nominal A sample support', 'Identify background samples controlling A yield and variance.'),
        ('boundary_migration', 'Disjoint boundary migration', 'Separate the isolation removal, angle removal and retained WP2 A.'),
        ('nominal_mc_distribution', 'Nominal MC boundary view', 'Show the local MC yield and effective-count support without nominal DATA.'),
        ('mass_closure', 'Closure by mass interval', 'Compare mass components and their union at exactly the same fixed WPs.'),
        ('mass_composition', 'Mass and process composition', 'Test low-mass dilution and changes in the background mixture.'),
        ('mass_mixing', 'Mass pooling and ABCD nonlinearity', 'Compare pooled ABCD prediction with the sum of slice predictions.'),
        ('nominal_c_contamination', 'Nominal C contamination', 'Keep the worst-model identities and the separate sideband purity checks.'),
        ('significance_comparison', 'MC significance with finite support', 'Compare known-background and finite-MC expected-sensitivity diagnostics.'),
        ('paired_wp_comparison', 'Same-model WP2 / WP1 comparison', 'Pair every matching signal hypothesis before comparing the WPs.'),
        ('final_safety', 'Final nominal DATA-access checks', 'Keep contamination and geometric protection distinct without accessing DATA.'),
    ]
    return finalize_final_check_plots_pd.DataFrame(entries, columns=['name', 'title', 'purpose'])


def finalize_final_check_plots_make_final_check_figures(result, config):
    """Build the final-check deck without loading or changing analysis inputs."""
    points = finalize_final_check_plots__table(result, 'working_points')
    if tuple(points.wp) != finalize_final_check_plots_WPS:
        raise ValueError('Final-check figures require Reference, WP1 and WP2 in order.')
    factories = (finalize_final_check_plots__summary, finalize_final_check_plots__nominal_support, finalize_final_check_plots__sample_leverage, finalize_final_check_plots__migration, finalize_final_check_plots__nominal_distribution,
                 finalize_final_check_plots__mass_closure, finalize_final_check_plots__mass_composition, finalize_final_check_plots__mass_mixing, finalize_final_check_plots__c_contamination,
                 finalize_final_check_plots__significance, finalize_final_check_plots__paired, finalize_final_check_plots__safety)
    definitions = finalize_final_check_plots_final_check_figure_definitions()
    with finalize_final_check_plots_plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.titlesize': 13, 'axes.labelsize': 11,
                         'figure.max_open_warning': 0}):
        figures = [(factory(result), name) for factory, name in zip(factories, definitions.name)]
    if len(figures) != len(definitions):
        raise AssertionError('Final-check figures differ from the declared figure definitions.')
    for figure, _ in figures:
        figure.canvas.draw()
        finalize_final_check_plots_plt.close(figure)
    return figures


# ========================================================================
# abcd_capped_gap_plots: original study API
# ========================================================================
"""Compact buffer comparisons with every isolation edge capped at 0.5.

Only validated MC and explicitly approved control tables are consumed.
"""
import numpy as finalize_capped_gap_plots_np
import pandas as finalize_capped_gap_plots_pd
import matplotlib.pyplot as finalize_capped_gap_plots_plt
from matplotlib.lines import Line2D as finalize_capped_gap_plots_Line2D

finalize_capped_gap_plots__canvas = _FINALIZE_NAMESPACES['abcd_final_check_plots']._canvas; finalize_capped_gap_plots__caption = _FINALIZE_NAMESPACES['abcd_final_check_plots']._caption; finalize_capped_gap_plots__style_table = _FINALIZE_NAMESPACES['abcd_final_check_plots']._style_table


def finalize_capped_gap_plots_capped_gap_figures(result):
    rows = result['tables']['sideband_scan']
    rows = rows.loc[rows.strategy.eq('gap') & rows.b_iso.le(.5)].copy()
    if len(rows) != 40 or not finalize_capped_gap_plots_np.allclose(rows.a_iso, .2) or not finalize_capped_gap_plots_np.allclose(rows.a_phi, 2.7):
        raise AssertionError('Require 40 native buffers with fixed WP2 A and isolation edges <=0.5.')
    chosen = rows.set_index('candidate_id',drop=False).loc[result['selections']['gap']['candidate_id']]
    eligible = rows.loc[rows.operational_pass]
    if not chosen.operational_pass or chosen.native_steps != eligible.native_steps.min():
        raise AssertionError('The selected buffer must minimize movement after MC purity and compatibility with both final A regions.')
    if not finalize_capped_gap_plots_np.allclose(rows.z_known_b_ratio_WP2_median, 1.) or not finalize_capped_gap_plots_np.allclose(rows.z_mc_stat_diagnostic_ratio_WP2_median, 1.):
        raise AssertionError('Every buffer must preserve the same MC-A sensitivity.')
    nominal = result['tables']['if_scan']
    reference = nominal.loc[finalize_capped_gap_plots_np.isclose(nominal.a_iso, .2) & finalize_capped_gap_plots_np.isclose(nominal.a_phi, 2.7)]
    if len(reference) != 1 or not finalize_capped_gap_plots_np.allclose(rows.z_known_b_median, reference.iloc[0].z_known_b_median):
        raise AssertionError('Buffer absolute significance must match the same WP2 A and signal ensemble.')
    return [(finalize_capped_gap_plots__gap_tradeoff(rows, chosen, reference.iloc[0]), 'gap_tradeoff'),
            (finalize_capped_gap_plots__gap_support(rows, chosen), 'gap_support'),
            (finalize_capped_gap_plots__gap_selection(rows, chosen, result['tables']['sbd_summary']), 'gap_selection')]


def finalize_capped_gap_plots_capped_gap_definitions():
    return finalize_capped_gap_plots_pd.DataFrame([
        ('gap_tradeoff', 'Buffer sensitivity and contamination with isolation edges <=0.5',
         'Show absolute MC-A sensitivity medians and changing B/C/D contamination on the capped grid.'),
        ('gap_support', 'Buffer control statistics and MC prediction',
         'Explain weighted yields, effective events and prediction precision as the isolation buffer changes.'),
        ('gap_selection', 'Selected minimum-change buffer and its controls',
         'Compare the purity minimum and final-A-compatible buffer with nominal DATA controls.')],
        columns=['name', 'title', 'purpose'])


def finalize_capped_gap_plots__gap_tradeoff(rows, chosen, reference):
    fig, axes = finalize_capped_gap_plots__canvas('Buffer scan: fixed WP2 A, isolation sideband edge at most 0.5',
                        cols=3, height=7.6, top=.83, bottom=.28)
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    for color, phi in zip(colors, [2.7, 2.4, 2., 1.5]):
        part = rows.loc[finalize_capped_gap_plots_np.isclose(rows.c_phi, phi)].sort_values('b_iso')
        axes[0, 0].plot(part.b_iso, finalize_capped_gap_plots_np.full(len(part), reference.s_sqrt_s_plus_b_median),
                       'o-', color=color, ms=4, label=f'C/D angle upper edge = {phi:g}')
        axes[0, 1].plot(part.b_iso, part.z_known_b_median, 'o-', color=color, ms=4)
        axes[0, 1].plot(part.b_iso, part.z_mc_stat_diagnostic_median, '--', color=color, alpha=.65)
        axes[0, 2].plot(part.b_iso, 100*part.max_S_B_BCD, 'o-', color=color, ms=5)
        good = part.purity_pass
        axes[0, 2].scatter(part.loc[good, 'b_iso'], 100*part.loc[good, 'max_S_B_BCD'],
                           s=110, facecolors='none', edgecolors='#248342', lw=1.6)
    axes[0, 0].set(title=r'$S/\sqrt{S+B}$ in MC A', ylabel='Median S / √(S+B) (absolute)',
                   ylim=(0, 1.2*reference.s_sqrt_s_plus_b_median))
    axes[0, 1].set(title='Asimov significance Z in MC A', ylabel='Median Z (absolute)',
                   ylim=(0, 1.2*reference.z_known_b_median))
    axes[0, 2].set(title='Nominal control contamination', ylabel='Maximum B/C/D S/B_MC [%]', yscale='log')
    axes[0, 2].axhline(10., color='#B32424', ls='--')
    for index, ax in enumerate(axes[0]):
        ax.set(xticks=[.2, .25, .3, .4, .5], xlim=(.185, .515), xlabel='B/D isolation lower edge I_H')
        value = [reference.s_sqrt_s_plus_b_median, chosen.z_known_b_median, 100*chosen.max_S_B_BCD][index]
        ax.scatter([chosen.b_iso], [value],
                   marker='*', s=200, color='#B32424', edgecolors='white', zorder=5)
    handles, _ = axes[0, 0].get_legend_handles_labels()
    handles += [finalize_capped_gap_plots_Line2D([], [], color='#444444', ls='--', label='Dashed Z: finite-MC diagnostic'),
                finalize_capped_gap_plots_Line2D([], [], marker='o', mfc='none', mec='#248342', ls='none', label='MC B/C/D contamination <10%'),
                finalize_capped_gap_plots_Line2D([], [], marker='*', color='#B32424', ls='none', label=f'Selected: I_H={chosen.b_iso:.2f}, phi_L={chosen.c_phi:g}')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .13), ncol=3, frameon=False, fontsize=9)
    finalize_capped_gap_plots__caption(fig, 'A is identical for every point: isolation<0.20 and |dphi|>=2.7. Signal and direct MC background in A do not change.\n'
             'Both sensitivity panels show absolute medians over the same 60 matching hypotheses, without normalization to WP2. Fixed A makes them constant across buffers.\n'
             'Right: maximum over 120 hypotheses and B/C/D; larger I_H excludes more isolation sideband. C/D phi_L is a sideband edge, not the A angle.\n'
             'The selected buffer minimizes movement while passing central purity and keeping both final A regions blinded; historical candidate A unions are not retained. Closure remains a separate check.')
    return fig


def finalize_capped_gap_plots__gap_support(rows, chosen):
    part = rows.loc[finalize_capped_gap_plots_np.isclose(rows.c_phi, chosen.c_phi)].sort_values('b_iso')
    fig, axes = finalize_capped_gap_plots__canvas(f'Buffer control statistics: move isolation sidebands at fixed C/D angle <{chosen.c_phi:g}',
                        rows=2, cols=2, height=8.8, top=.85, bottom=.23)
    colors = ['#B32424', '#0072B2', '#D55E00', '#009E73']
    for region, color in zip('ABCD', colors):
        axes[0, 0].errorbar(part.b_iso, part[region], yerr=part[region+'_error'], fmt='o-',
                            color=color, ms=4, capsize=2, label=region)
        axes[0, 1].plot(part.b_iso, part['neff_'+region], 'o-', color=color, ms=4, label=region)
    for region, color in zip('BCD', colors[1:]):
        axes[1, 0].plot(part.b_iso, 100*part['max_S_B_'+region], 'o-', color=color, ms=4, label=region)
    axes[1, 0].axhline(10, ls='--', color='#B32424')
    axes[1, 1].errorbar(part.b_iso, part.A, yerr=part.A_error, fmt='o-', color='#B32424', label='Direct MC A')
    axes[1, 1].errorbar(part.b_iso, part.pred, yerr=part.pred_error, fmt='s--', color='#0072B2', label='MC B×C/D')
    axes[0, 0].set(title='Weighted background yields', ylabel='Sum of MC weights')
    axes[0, 1].set(title='Effective MC sample size', ylabel='N_eff = (sum w)² / sum w²', yscale='log')
    axes[1, 0].set(title='Contamination by control region', ylabel='Maximum S/B_MC [%]', yscale='log')
    axes[1, 1].set(title='Does the MC prediction match direct MC A?', ylabel='Weighted MC events')
    for ax in axes.flat:
        ax.axvline(chosen.b_iso, color='#B32424', ls=':', lw=1.4)
        ax.set(xticks=[.2, .25, .3, .4, .5], xlim=(.185, .515), xlabel='B/D isolation lower edge I_H')
        ax.legend(frameon=False, fontsize=9)
    finalize_capped_gap_plots__caption(fig, f'All panels use phi_L={chosen.c_phi:g} and the same fixed WP2 A. The red vertical line is I_H={chosen.b_iso:.2f}.\n'
             f'N_eff is a weighted-statistics diagnostic, not a raw count. C is unchanged by moving I_H and has {chosen.neff_C:.2f} effective MC events.\n'
             'The large MC prediction uncertainty and low MC closure require care; no nominal DATA A is used in this figure.')
    return fig


def finalize_capped_gap_plots__gap_selection(rows, chosen, controls):
    near = rows.loc[finalize_capped_gap_plots_np.isclose(rows.c_phi,chosen.c_phi)&rows.b_iso.le(.4)].sort_values('b_iso')
    pure_min = rows.loc[rows.purity_pass].sort_values(['native_steps','b_iso','c_phi'],ascending=[True,True,False]).iloc[0]
    data = controls.loc[controls.candidate_id.eq(chosen.candidate_id)].iloc[0]
    fig, axes = finalize_capped_gap_plots__canvas('Selected buffer: minimum movement, supported controls, and remaining limitations',
                        cols=1, height=7.8, top=.85, bottom=.30)
    axes[0, 0].axis('off')
    contents = [[f'{r.b_iso:g}', f'{r.c_phi:g}', f'{r.native_steps:.0f}', f'{100*r.max_S_B_BCD:.3f}%',
                 f'{r.min_sideband_neff:.2f}', f'{r.pred:.2f} ± {r.pred_error:.2f}',
                 'YES' if r.finalist_A_compatible else 'NO'] for r in near.itertuples()]
    tab = axes[0, 0].table(cellText=contents,
        colLabels=['I_H', 'phi_L', 'Native movement', 'Max BCD S/B_MC', 'Min BCD N_eff', 'MC B×C/D', 'Both final A blind'],
        cellLoc='center', colWidths=[.08, .08, .15, .18, .15, .21, .15], bbox=[0, .30, 1, .65])
    finalize_capped_gap_plots__style_table(tab, 10)
    selected_row=int(finalize_capped_gap_plots_np.flatnonzero(near.candidate_id.eq(chosen.candidate_id).to_numpy())[0])+1
    for j in range(7):
        tab[(selected_row, j)].set_facecolor('#FFF0DC')
    fig.text(.075, .37, f'Selected: A remains isolation<{chosen.a_iso:.2f}, |dphi|≥{chosen.a_phi:g}; B/D isolation≥{chosen.b_iso:.2f}; C/D |dphi|<{chosen.c_phi:g}.\n'
             f'Nominal DATA B/C/D = {data.data_B:.0f}/{data.data_C:.0f}/{data.data_D:.0f}; raw B×C/D = {data.pred:.2f} ± {data.pred_error:.2f} (counting only).',
             fontsize=12, weight='bold', color='#25354A')
    fig.text(.075, .24, f'The purity-only minimum is (I_H, phi_L)=({pure_min.b_iso:.2f},{pure_min.c_phi:g}); its B overlaps the selected loose A.\n'
             f'I_H={chosen.b_iso:.2f} is the smallest edge allowing nominal B/C/D DATA comparison while both current final A regions remain blind.',
             fontsize=11, color='#8D3429')
    finalize_capped_gap_plots__caption(fig, 'All these buffers preserve the same MC-A sensitivity. The choice concerns sideband support and compatibility of the two current final candidates, not historical A unions.\n'
             'C MC remains sparse and near the central 10% threshold. Nominal DATA A is blinded; closure, transfer uncertainty and expected limits require their full statistical model.')
    return fig


# ========================================================================
# abcd_sideband_plots: original study API
# ========================================================================
"""Meeting figures for loosened boundaries and WP2-A-fixed sidebands.

Only validated calculation tables are accepted. No original DATA is opened,
no release decision is changed, and no new working point is ranked here.
"""
import textwrap as finalize_sideband_plots_textwrap

import matplotlib.pyplot as finalize_sideband_plots_plt
from matplotlib.colors import ListedColormap as finalize_sideband_plots_ListedColormap; from matplotlib.colors import LogNorm as finalize_sideband_plots_LogNorm
from matplotlib.lines import Line2D as finalize_sideband_plots_Line2D
from matplotlib.patches import Patch as finalize_sideband_plots_Patch; from matplotlib.patches import Rectangle as finalize_sideband_plots_Rectangle
import numpy as finalize_sideband_plots_np
import pandas as finalize_sideband_plots_pd

finalize_sideband_plots__canvas = _FINALIZE_NAMESPACES['abcd_final_check_plots']._canvas; finalize_sideband_plots__caption = _FINALIZE_NAMESPACES['abcd_final_check_plots']._caption; finalize_sideband_plots__style_table = _FINALIZE_NAMESPACES['abcd_final_check_plots']._style_table


finalize_sideband_plots_SCOPES = ('VR1_DD', 'VR1_PP', 'VR1_PD', 'VR1_DP', 'VR2_PP', 'VR2_PD', 'VR2_DP')
finalize_sideband_plots_SCOPE_LABELS = {'VR1_DD': 'Low mass · DD', 'VR1_PP': 'Low mass · PP',
                'VR1_PD': 'Low mass · PD', 'VR1_DP': 'Low mass · DP',
                'VR2_PP': 'Intermediate · PP', 'VR2_PD': 'Intermediate · PD',
                'VR2_DP': 'Intermediate · DP'}
finalize_sideband_plots_STRATEGY_COLORS = {'baseline': '#777777', 'loose': '#0072B2', 'gap': '#D55E00'}


def finalize_sideband_plots__table(result, name):
    return result['tables'][name].copy()


def finalize_sideband_plots__scan(result):
    rows = finalize_sideband_plots__table(result, 'sideband_scan')
    required = {'candidate_id', 'strategy', 'a_iso', 'a_phi', 'b_iso', 'c_phi',
                'max_S_B_BCD', 'purity_pass', 'geometry_safe', 'operational_pass',
                'min_sideband_neff', 'pred', 'pred_error',
                'z_known_b_ratio_WP2_median', 'z_mc_stat_diagnostic_ratio_WP2_median'}
    if not required.issubset(rows):
        raise ValueError('Missing sideband plot fields: ' + ', '.join(sorted(required - set(rows))))
    rows['prediction_rel_error'] = finalize_sideband_plots_np.divide(rows.pred_error, rows.pred,
                                            out=finalize_sideband_plots_np.full(len(rows), finalize_sideband_plots_np.nan), where=rows.pred > 0)
    return rows


def finalize_sideband_plots__selected(result):
    chosen = finalize_sideband_plots__table(result, 'sideband_selection')
    ids = chosen.candidate_id.tolist()
    if len(ids) != len(set(ids)):
        raise AssertionError('Each displayed candidate must have a unique selection record.')
    rows = finalize_sideband_plots__scan(result).set_index('candidate_id').loc[ids].reset_index()
    if not len(rows):
        raise ValueError('The meeting figures require at least the WP2 baseline.')
    return rows


def finalize_sideband_plots__label(row, newline=False):
    if row['strategy'] == 'baseline':
        return str(row['candidate_id']).replace('baseline_', '')
    prefix = 'L' if row['strategy'] == 'loose' else 'G'
    x = row['a_iso'] if prefix == 'L' else row['b_iso']
    y = row['a_phi'] if prefix == 'L' else row['c_phi']
    return f'{prefix}' + ('\n' if newline else ': ') + f'{x:g}/{y:g}'


def finalize_sideband_plots__colors(rows):
    palette = {'baseline': ['#777777', '#222222', '#999999'],
               'loose': ['#0072B2', '#56B4E9', '#005087', '#76A5AF'],
               'gap': ['#D55E00', '#E69F00', '#A63E00', '#A577B5']}
    used, colors = {name: 0 for name in palette}, {}
    for _, row in rows.iterrows():
        strategy = row['strategy']
        colors[row.candidate_id] = palette[strategy][used[strategy] % len(palette[strategy])]
        used[strategy] += 1
    return colors


def finalize_sideband_plots__data_mask(frame):
    data = frame.loc[frame.source.eq('DATA')]
    required = {'approved', 'display_allowed', 'reason'}
    if not required.issubset(data) or data[list(required - {'reason'})].isna().any().any():
        raise AssertionError('Every VR DATA row needs explicit purity and display decisions.')
    fields = [name for name in data if name in {
        'A', 'B', 'C', 'D', 'pred', 'pred_error', 'kappa', 'kappa_error', 'pull',
        'A_error', 'B_error', 'C_error', 'D_error', 'min_neff', 'min_sideband_neff',
        'R_high', 'R_high_error', 'R_low', 'R_low_error'} or name.startswith(('var_', 'neff_'))]
    if not data.loc[~data.display_allowed.astype(bool), fields].isna().all().all():
        raise AssertionError('A withheld sideband-VR DATA measurement is not masked.')
    visible = data.loc[data.display_allowed.astype(bool)]
    if not visible.approved.astype(bool).all():
        raise AssertionError('A displayed VR DATA measurement failed its new integrated gate.')
    if 'signal_hypotheses' in visible and not visible.signal_hypotheses.eq(120).all():
        raise AssertionError('Every displayed new DATA selection needs all 120 signal hypotheses.')
    if 'max_S_B' in visible:
        ratios = visible.max_S_B.to_numpy(float)
        if not (finalize_sideband_plots_np.isfinite(ratios) & (ratios >= 0) & (ratios < .1)).all():
            raise AssertionError('A displayed DATA selection violates the strict 10% threshold.')
    return visible


def finalize_sideband_plots__approaches(result):
    fig, axes = finalize_sideband_plots__canvas('Two approaches: change A, or keep A and separate the sidebands',
                       cols=2, height=8.9, bottom=.25, top=.85)
    for ax in axes[0]:
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Muon-LJ isolation', ylabel='|Δφ|',
               xticks=[], yticks=[])
        ax.grid(False)
    left, right = axes[0]
    x, y = .49, .41
    for name, xx, yy, width, height, color in [
            ('A', 0, y, x, 1-y, '#FBE5E5'), ('B', x, y, 1-x, 1-y, '#E8EFF8'),
            ('C', 0, 0, x, y, '#E8EFF8'), ('D', x, 0, 1-x, y, '#E8EFF8')]:
        left.add_patch(finalize_sideband_plots_Rectangle((xx, yy), width, height, facecolor=color, edgecolor='#8795A7'))
        left.text(xx+width/2, yy+height/2, name, ha='center', va='center', fontsize=25, weight='bold')
    left.add_patch(finalize_sideband_plots_Rectangle((0, .70), .30, .30, fill=False, edgecolor='#D55E00', ls='--', lw=2))
    left.text(.15, .84, 'Original WP2 A', ha='center', fontsize=9, color='#9F4200')
    left.annotate('', xy=(.45, .46), xytext=(.31, .69),
                  arrowprops=dict(arrowstyle='->', lw=2, color='#9F2929'))
    left.set_title('Loose standard boundary: A expands', fontsize=13)
    left.text(.5, -.09, 'I_A ≥ 0.20; φ_A ≤ 2.7\nB/C/D share the same two new boundaries.',
              transform=left.transAxes, ha='center', va='top', fontsize=10)
    right.set_facecolor('#eeeeee')
    xlo, xhi, ylo, yhi = .30, .55, .40, .70
    for name, xx, yy, width, height, color in [
            ('A', 0, yhi, xlo, 1-yhi, '#FBE5E5'), ('B', xhi, yhi, 1-xhi, 1-yhi, '#E8EFF8'),
            ('C', 0, 0, xlo, ylo, '#E8EFF8'), ('D', xhi, 0, 1-xhi, ylo, '#E8EFF8')]:
        right.add_patch(finalize_sideband_plots_Rectangle((xx, yy), width, height, facecolor=color, edgecolor='#8795A7'))
        right.text(xx+width/2, yy+height/2, name, ha='center', va='center', fontsize=25, weight='bold')
    right.text(.15, .76, 'WP2 unchanged', fontsize=8.5, ha='center', color='#9F2929')
    right.text(.425, .55, 'buffer', ha='center', va='center', fontsize=13, color='#666666', rotation=90)
    right.text(.77, .55, 'buffer', ha='center', va='center', fontsize=13, color='#666666')
    right.set_title('Gap: WP2 A fixed; B/C/D move away', fontsize=13)
    right.text(.5, -.09, 'A: I<0.20, φ≥2.7; B: I≥I_H, φ≥2.7\nC: I<0.20, φ<φ_L; D: I≥I_H, φ<φ_L',
               transform=right.transAxes, ha='center', va='top', fontsize=10)
    finalize_sideband_plots__caption(fig, 'Schematic coordinates only. Nominal DD, MJJ≥150 GeV and partner isolation [0,0.5) are unchanged. No nominal DATA value is shown.\n'
             'The gap retains exactly the same MC signal and background in A, so its direct-MC A sensitivity equals WP2. Sideband statistics and ABCD closure can still change.\n'
             'The unchanged protected-A union requires I_H≥0.30 and φ_L≤2.4 for these rectangular gap sidebands; purity must be checked separately.\n'
             'Each new definition is applied unchanged in the validation regions. A buffer does not inherit the old closure validation automatically.')
    return fig


def finalize_sideband_plots__heatmap(ax, rows, xfield, yfield, metric, title, transform=1., log=False,
             fmt='.2f', selected=()):
    x, y = sorted(rows[xfield].unique()), sorted(rows[yfield].unique())
    if rows.duplicated([xfield, yfield]).any() or len(rows) != len(x)*len(y):
        raise AssertionError('A sideband scan display requires a unique rectangular native-boundary grid.')
    values = rows.pivot(index=yfield, columns=xfield, values=metric).reindex(index=y, columns=x).to_numpy(float) * transform
    finite = finalize_sideband_plots_np.isfinite(values)
    view = values.copy()
    norm = None
    if log and finite.any():
        positive = values[finite & (values > 0)]
        if len(positive):
            vmin, vmax = max(float(positive.min()), 1e-6), float(positive.max())
            vmax = max(vmax, vmin*1.01)
            view[finite] = finalize_sideband_plots_np.maximum(view[finite], vmin)
            norm = finalize_sideband_plots_LogNorm(vmin=vmin, vmax=vmax)
    cmap = finalize_sideband_plots_plt.get_cmap('viridis').copy()
    cmap.set_bad('#dddddd')
    artist = ax.imshow(finalize_sideband_plots_np.ma.masked_invalid(view), origin='lower', aspect='auto', cmap=cmap, norm=norm)
    for iy in range(len(y)):
        for ix in range(len(x)):
            label = format(values[iy, ix], fmt) if finite[iy, ix] else '—'
            ax.text(ix, iy, label, color='white' if finite[iy, ix] else '#555555',
                    ha='center', va='center', fontsize=7.6,
                    bbox=dict(facecolor='black', alpha=.23, edgecolor='none', pad=.4))
    for _, row in rows.iterrows():
        ix, iy = x.index(row[xfield]), y.index(row[yfield])
        if row.operational_pass:
            ax.add_patch(finalize_sideband_plots_Rectangle((ix-.48, iy-.48), .96, .96, fill=False, edgecolor='#58DB86', lw=1.5))
        if row.candidate_id in selected:
            ax.add_patch(finalize_sideband_plots_Rectangle((ix-.41, iy-.41), .82, .82, fill=False, edgecolor='#F9BE35', lw=2))
    ax.set(xticks=finalize_sideband_plots_np.arange(len(x)), xticklabels=[f'{v:g}' for v in x],
           yticks=finalize_sideband_plots_np.arange(len(y)), yticklabels=[f'{v:g}' for v in y], title=title)
    ax.grid(False)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    return artist


def finalize_sideband_plots__scan_figure(result, strategy):
    rows = finalize_sideband_plots__scan(result).query('strategy == @strategy')
    chosen = finalize_sideband_plots__selected(result)
    ids = chosen.candidate_id.tolist()
    if strategy == 'loose':
        title, xf, yf = 'Looser boundaries: sensitivity versus nominal B/C/D contamination', 'a_iso', 'a_phi'
        metrics = [
            ('z_known_b_ratio_WP2_median', 'Median model-wise Z / WP2', 1., False, '.2f'),
            ('max_S_B_BCD', 'Maximum S/B in B/C/D [%]', 100., True, '.1f'),
            ('min_sideband_neff', 'Minimum B/C/D effective MC count', 1., True, '.1f')]
    else:
        title, xf, yf = 'Fixed WP2 A: what changes when a buffer is added?', 'b_iso', 'c_phi'
        metrics = [
            ('max_S_B_BCD', 'Maximum S/B in B/C/D [%]', 100., True, '.1f'),
            ('min_sideband_neff', 'Minimum B/C/D effective MC count', 1., True, '.1f'),
            ('prediction_rel_error', 'MC σ(BC/D) / (BC/D)', 1., True, '.2f')]
    fig, axes = finalize_sideband_plots__canvas(title, cols=3, height=9.2, bottom=.25, top=.86)
    for ax, (metric, label, scale, log, fmt) in zip(axes[0], metrics):
        art = finalize_sideband_plots__heatmap(ax, rows, xf, yf, metric, label, scale, log, fmt, ids)
        fig.colorbar(art, ax=ax, fraction=.047, pad=.025)
        ax.set_xlabel('Isolation boundary for A/B' if strategy == 'loose' else 'High-isolation sideband edge I_H')
        ax.set_ylabel('Angle boundary for A/C' if strategy == 'loose' else 'Low-angle sideband edge φ_L')
    handles = [finalize_sideband_plots_Patch(facecolor='none', edgecolor='#58DB86', lw=2, label='Nominal BCD purity + protected geometry pass'),
               finalize_sideband_plots_Patch(facecolor='none', edgecolor='#F9BE35', lw=2, label='Shortlisted for further checks')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .13), ncol=2, frameon=False, fontsize=9)
    if strategy == 'loose':
        text = ('Loosening expands A relative to WP2. Signal sensitivity uses each matching model separately and the direct nominal MC A background.\n'
                'All 120 hypotheses enter the strict B/C/D S/B<10% gate; A contamination is not a nominal sideband criterion. Geometry is checked independently.\n')
    else:
        text = ('Every gap point keeps the same WP2 A and therefore the same direct-MC A sensitivity; there is no new sensitivity gain from moving sidebands.\n'
                'Purity, remaining MC support and prediction precision are separate criteria. The unchanged protected geometry requires I_H≥0.30 and φ_L≤2.4.\n')
    finalize_sideband_plots__caption(fig, text + 'Equal-sized cells enumerate native boundary choices; blank cells indicate undefined metrics. Positive MC support and low-count uncertainty treatment remain essential.\n'
             'Green outlines are MC feasibility checks only, not DATA-access or final-background-model approval. No DATA is used to rank these candidates.')
    return fig


def finalize_sideband_plots__selected_support(result):
    rows = finalize_sideband_plots__selected(result)
    colors = finalize_sideband_plots__colors(rows)
    fig, axes = finalize_sideband_plots__canvas('Shortlisted definitions: nominal MC support and contamination',
                       rows=2, cols=2, height=10.1, bottom=.22, top=.87)
    count = len(rows)
    offsets = finalize_sideband_plots_np.linspace(-.31, .31, count) if count > 1 else [0.]
    for (_, row), offset in zip(rows.iterrows(), offsets):
        color, label = colors[row.candidate_id], finalize_sideband_plots__label(row)
        values = finalize_sideband_plots_np.array([row[r] for r in 'ABCD'], float)
        errors = finalize_sideband_plots_np.array([row[r+'_error'] for r in 'ABCD'], float)
        neff = finalize_sideband_plots_np.array([row['neff_'+r] for r in 'ABCD'], float)
        axes[0, 0].errorbar(finalize_sideband_plots_np.arange(4)+offset, values, yerr=errors, fmt='o', color=color,
                            capsize=2, markersize=4, label=label)
        axes[0, 1].plot(finalize_sideband_plots_np.arange(4)+offset, neff, 'o', color=color, markersize=4)
    axes[0, 0].set(yscale='symlog', ylabel='MC yield ± √sumw²', xticks=finalize_sideband_plots_np.arange(4), xticklabels=list('ABCD'))
    axes[0, 1].set(yscale='log', ylabel='Effective MC count N_eff', xticks=finalize_sideband_plots_np.arange(4), xticklabels=list('ABCD'))
    axes[0, 1].axhline(10, color='#999999', ls=':', lw=1)
    regions = finalize_sideband_plots__table(result, 'sideband_regions')
    for (_, row), offset in zip(rows.iterrows(), offsets):
        regional = regions.loc[regions.candidate_id.eq(row.candidate_id)].set_index('region').reindex(list('BCD'))
        axes[1, 0].plot(finalize_sideband_plots_np.arange(3)+offset, 100*regional.max_S_B, 'o',
                        color=colors[row.candidate_id], markersize=4)
    axes[1, 0].set(yscale='log', ylabel='Maximum S/B across 120 hypotheses [%]',
                   xticks=finalize_sideband_plots_np.arange(3), xticklabels=list('BCD'))
    axes[1, 0].axhline(10, color='#A33333', ls='--', lw=1)
    x = finalize_sideband_plots_np.arange(count)
    axes[1, 1].errorbar(x-.08, rows.A, yerr=rows.A_error, fmt='o', color='#222222', capsize=3, label='Direct MC A')
    axes[1, 1].errorbar(x+.08, rows.pred, yerr=rows.pred_error, fmt='s', markerfacecolor='white',
                        color='#E69F00', capsize=3, label='MC B×C/D')
    axes[1, 1].set(yscale='symlog', ylabel='MC yield / prediction',
                   xticks=x, xticklabels=[finalize_sideband_plots__label(row, True) for _, row in rows.iterrows()])
    axes[1, 1].tick_params(axis='x', labelsize=8)
    axes[1, 1].legend(fontsize=8)
    fig.legend(handles=[finalize_sideband_plots_Line2D([0], [0], color=colors[row.candidate_id], marker='o', ls='none', label=finalize_sideband_plots__label(row))
                        for _, row in rows.iterrows()], loc='lower center', bbox_to_anchor=(.5, .13),
               ncol=min(count, 4), frameon=False, fontsize=8)
    finalize_sideband_plots__caption(fig, 'Nominal MC only. L labels list the common isolation/angle boundary; G labels list I_H/φ_L, with WP2 A held fixed.\n'
             'N_eff measures weighted support, not raw event counts. The line at N_eff=10 is a diagnostic guide, not a new acceptance condition.\n'
             'B×C/D errors use disjoint-region first-order sumw² propagation; sparse ratio tails are not Gaussian confidence intervals. Passing contamination alone does not establish closure.')
    return fig


def finalize_sideband_plots__closure_points(ax, x, rows, color, marker, label, filled):
    values, errors = rows.kappa.to_numpy(float), rows.kappa_error.to_numpy(float)
    good = finalize_sideband_plots_np.isfinite(values) & finalize_sideband_plots_np.isfinite(errors) & (errors >= 0)
    ax.errorbar(x[good], values[good], yerr=errors[good], fmt=marker, color=color,
                markerfacecolor=color if filled else 'white', markersize=4, capsize=2,
                linestyle='none', label=label)
    outside = good & ((values < 0) | (values > 4))
    for index in finalize_sideband_plots_np.flatnonzero(outside):
        upper = values[index] > 4
        y = 3.88 if upper else .12
        ax.plot(x[index], y, '^' if upper else 'v', color=color, markersize=5)
        ax.text(x[index], 3.45 if upper else .5, f'{values[index]:.2g}', color=color, fontsize=7, ha='center')
    return int(finalize_sideband_plots_np.count_nonzero(good & ((values-errors < 0) | (values+errors > 4))))


def finalize_sideband_plots__vr_closure(result):
    rows = finalize_sideband_plots__selected(result)
    frame = finalize_sideband_plots__table(result, 'sbvr_closure')
    finalize_sideband_plots__data_mask(frame)
    ids = rows.candidate_id.tolist()
    fig, axes = finalize_sideband_plots__canvas('Identical sideband definitions in the validation regions',
                       rows=2, cols=4, height=10.5, bottom=.22, top=.87)
    x = finalize_sideband_plots_np.arange(len(ids))
    labels = [finalize_sideband_plots__label(row, True) for _, row in rows.iterrows()]
    for ax, scope in zip(axes.ravel(), finalize_sideband_plots_SCOPES):
        selected = frame.loc[frame.scope.eq(scope)]
        for source, color, marker, shift, filled in [('MC', '#E69F00', 's', -.08, False),
                                                     ('DATA', '#222222', 'o', .08, True)]:
            table = selected.loc[selected.source.eq(source)].set_index('candidate_id').reindex(ids)
            n_clipped = finalize_sideband_plots__closure_points(ax, x+shift, table, color, marker, source, filled)
            if source == 'DATA':
                blocked = ~table.display_allowed.fillna(False).astype(bool).to_numpy()
                ax.scatter(x[blocked]+shift, finalize_sideband_plots_np.full(blocked.sum(), -.035), marker='x', color='#777777',
                           transform=ax.get_xaxis_transform(), clip_on=False, s=20)
                ax.text(.03, .96, f'DATA: {int((~blocked).sum())}/{len(ids)} shown', transform=ax.transAxes,
                        va='top', fontsize=7, bbox=dict(facecolor='white', alpha=.85, edgecolor='none', pad=.5))
            elif n_clipped:
                ax.text(.03, .89, f'MC errors clipped: {n_clipped}', transform=ax.transAxes,
                        va='top', fontsize=6.7, color='#666666')
        ax.set(ylim=(0, 4), xticks=x, xticklabels=labels, title=finalize_sideband_plots_SCOPE_LABELS[scope], ylabel='κ = A / (B×C/D)')
        ax.tick_params(axis='x', labelsize=6.8, pad=10)
        ax.axhline(1, color='#5F775B', ls='--', lw=1)
        ax.axhspan(.9, 1.1, color='#83B881', alpha=.13)
    ax = axes.ravel()[-1]
    ax.axis('off')
    handles = [finalize_sideband_plots_Line2D([0], [0], marker='s', markerfacecolor='white', color='#E69F00', ls='none', label='Background MC'),
               finalize_sideband_plots_Line2D([0], [0], marker='o', color='#222222', ls='none', label='Display-allowed DATA'),
               finalize_sideband_plots_Line2D([0], [0], marker='x', color='#777777', ls='none', label='DATA withheld')]
    ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=9)
    ax.text(.03, .53, 'Low: 0≤MJJ<150 GeV\nIntermediate: 150≤MJJ<300 GeV\n\nDD is tested only at low mass.\nNo nominal DATA are shown.',
            transform=ax.transAxes, fontsize=10, va='top')
    finalize_sideband_plots__caption(fig, 'Every new VR uses the exact nominal candidate boundaries. DATA require the all-120-hypothesis ABCD gate, available approved cache values and complementary-disclosure protection.\n'
             'A withheld point is not a zero measurement or a failed closure result. Plot range is κ∈[0,4]; off-frame central values are labelled with arrows and clipped MC error ranges are counted.\n'
             'The WPs share events. These diagnostic error bars do not provide independent Gaussian ranking tests or a closure uncertainty transferable to nominal DD.')
    return fig


def finalize_sideband_plots__release_code(row):
    if bool(row.display_allowed):
        return 0, 'SHOW'
    reason = str(row.reason).upper()
    if not bool(row.approved):
        return 1, 'GATE'
    if any(word in reason for word in ('COMPLEMENT', 'BUFFER', 'SUBTRACT', 'DISCLOS', 'ATOMIC')):
        return 3, 'BUFFER'
    if 'CACHE' in reason:
        return 2, 'CACHE'
    return 4, 'HOLD'


def finalize_sideband_plots__release_audit(result):
    rows = finalize_sideband_plots__selected(result)
    frame = finalize_sideband_plots__table(result, 'sbvr_closure')
    finalize_sideband_plots__data_mask(frame)
    data = frame.loc[frame.source.eq('DATA')]
    values = finalize_sideband_plots_np.full((len(rows), len(finalize_sideband_plots_SCOPES)), 4.)
    text = finalize_sideband_plots_np.full(values.shape, 'HOLD', object)
    for i, candidate in enumerate(rows.candidate_id):
        for j, scope in enumerate(finalize_sideband_plots_SCOPES):
            selected = data.loc[data.candidate_id.eq(candidate) & data.scope.eq(scope)]
            if len(selected) != 1:
                raise AssertionError('Expected one DATA release decision per candidate and VR.')
            values[i, j], text[i, j] = finalize_sideband_plots__release_code(selected.iloc[0])
    fig, axes = finalize_sideband_plots__canvas('Which new VR measurements can be displayed?', cols=1,
                       height=max(7.6, 5.4+len(rows)*.35), bottom=.29, top=.85)
    ax = axes[0, 0]
    cmap = finalize_sideband_plots_ListedColormap(['#C6E4CC', '#F3C6C6', '#D6C9E8', '#F6D4A5', '#D8DCE1'])
    ax.imshow(values, cmap=cmap, vmin=-.5, vmax=4.5, aspect='auto')
    for i in range(len(rows)):
        for j in range(len(finalize_sideband_plots_SCOPES)):
            ax.text(j, i, text[i, j], ha='center', va='center', fontsize=10, weight='bold', color='#293544')
    ax.set(xticks=finalize_sideband_plots_np.arange(len(finalize_sideband_plots_SCOPES)), xticklabels=[finalize_sideband_plots_SCOPE_LABELS[s].replace(' · ', '\n') for s in finalize_sideband_plots_SCOPES],
           yticks=finalize_sideband_plots_np.arange(len(rows)), yticklabels=[finalize_sideband_plots__label(r) for _, r in rows.iterrows()])
    ax.grid(False)
    fig.subplots_adjust(left=.17)
    handles = [finalize_sideband_plots_Patch(facecolor=color, label=label) for color, label in zip(cmap.colors,
                ['SHOW: all release checks pass', 'GATE: integrated MC purity/support fails',
                 'CACHE: required source DATA not approved', 'BUFFER: complement protection', 'HOLD: not released'])]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, .14), ncol=2, frameon=False, fontsize=9)
    finalize_sideband_plots__caption(fig, 'The new integrated VR purity gate is necessary but not sufficient for displaying DATA. Existing visible parents or neighboring WPs can expose an unsafe buffer by subtraction.\n'
             'Release decisions are made in the calculation helper before plotting. Hidden numeric DATA statistics stay NaN; this panel shows decisions only.\n'
             'This panel concerns VR DATA only. Nominal B/C/D require their own purity, protected-geometry and disclosure checks; nominal A stays blind.')
    return fig


def finalize_sideband_plots__comparison(result):
    rows = finalize_sideband_plots__selected(result)
    vr = finalize_sideband_plots__table(result, 'sbvr_closure')
    finalize_sideband_plots__data_mask(vr)
    counts = vr.loc[vr.source.eq('DATA')].groupby('candidate_id').display_allowed.sum()
    contents = []
    for _, row in rows.iterrows():
        contents.append([
            finalize_sideband_plots__label(row), f'{row.A:.3g} ± {row.A_error:.2g}', f'{row.signal_retention_WP2_median:.3f}',
            f'{row.z_known_b_ratio_WP2_median:.3f} / {row.z_mc_stat_diagnostic_ratio_WP2_median:.3f}',
            f'{row.max_S_B_BCD:.1%}', f'{row.min_sideband_neff:.2f}',
            'PASS' if row.geometry_safe else 'OVERLAP',
            f'{int(counts.get(row.candidate_id, 0))}/7'])
    fig, axes = finalize_sideband_plots__canvas('Decision evidence: sensitivity, purity, support and validation',
                       cols=1, height=max(9., 6.8+len(rows)*.25), bottom=.37, top=.86)
    ax = axes[0, 0]
    ax.axis('off')
    table = ax.table(cellText=contents, colLabels=['Definition', 'Nominal MC A',
        'Signal / WP2\nmedian', 'Z / WP2 medians\nknown-b / finite-MC',
        'Max BCD\nS/B', 'Min BCD\nN_eff', 'Protected\ngeometry', 'VR DATA\nshown'],
        cellLoc='center', colWidths=[.13, .16, .11, .18, .11, .10, .12, .09], bbox=[0, 0, 1, 1])
    finalize_sideband_plots__style_table(table, 9)
    scan = finalize_sideband_plots__scan(result)
    statements = []
    for strategy, label in [('loose', 'Looser boundary'), ('gap', 'WP2-A-fixed gap')]:
        part = scan.loc[scan.strategy.eq(strategy)]
        statements.append(f'{label}: {int(part.purity_pass.sum())}/{len(part)} pass nominal BCD purity; '
                          f'{int(part.operational_pass.sum())}/{len(part)} also avoid the protected A union.')
    fig.text(.08, .28, '\n'.join(statements), fontsize=11, va='top', color='#25354A')
    fig.text(.08, .19, 'Loose boundaries trade signal-region background against sensitivity. Gaps retain WP2 A but can reduce sideband statistics\nand expose contaminated buffers through existing DATA differences. VR closure and release checks remain separate.',
             fontsize=10.5, va='top', color='#25354A')
    finalize_sideband_plots__caption(fig, 'L: common isolation/angle boundary. G: high-isolation sideband edge / low-angle sideband edge, keeping WP2 A=(I<0.20, |Δφ|≥2.7).\n'
             'Sensitivity ratios compare the same 60 matching signal models; contamination uses all 120 models. Finite-MC Z is a direct-MC diagnostic, not a full ABCD limit calculation.\n'
             'No final sensitivity improvement is inferred solely from unchanged A or smaller contamination. Nominal A remains blind; a zero VR-display count does not establish failed closure.')
    return fig


def finalize_sideband_plots_sideband_figure_definitions():
    return finalize_sideband_plots_pd.DataFrame([
        ('sb_approaches', 'Loose boundaries and fixed-A buffers', 'Define the two approaches and their distinct effects on A.'),
        ('sb_loose_tradeoff', 'Loose-boundary MC tradeoff', 'Compare nominal MC sensitivity, BCD purity and sideband support.'),
        ('sb_gap_tradeoff', 'Fixed-A gap scan', 'Compare sideband purity and precision while holding direct-MC A sensitivity fixed.'),
        ('sb_selected_support', 'Shortlisted nominal support', 'Inspect ABCD yields, effective counts, contamination and MC closure.'),
        ('sb_vr_closure', 'Same-definition VR closure', 'Compare gated DATA and MC in seven explicit validation regions.'),
        ('sb_release_audit', 'New VR release decisions', 'Show purity, source-cache and complementary-protection restrictions.'),
        ('sb_comparison', 'Comparison for the meeting decision', 'Summarize the candidate tradeoffs and remaining validation scope.'),
        ('sb_nominal_sideband_data', 'Approved nominal B/C/D statistics', 'Show only released sideband counts and the corresponding uncorrected ABCD prediction.'),
        ('sb_meeting_conclusion', 'Proposal and unresolved transfer uncertainty', 'Compare the leading fixed-A buffer with the safe relaxed boundary and state the next decision.'),
    ], columns=['name', 'title', 'purpose'])


def finalize_sideband_plots_make_sideband_figures(result, config):
    """Return nine figures from validated nominal, VR and DATA-release tables."""
    finalize_sideband_plots__scan(result)
    finalize_sideband_plots__data_mask(finalize_sideband_plots__table(result, 'sbvr_closure'))
    builders = [('sb_approaches', finalize_sideband_plots__approaches),
                ('sb_loose_tradeoff', lambda r: finalize_sideband_plots__scan_figure(r, 'loose')),
                ('sb_gap_tradeoff', lambda r: finalize_sideband_plots__scan_figure(r, 'gap')),
                ('sb_selected_support', finalize_sideband_plots__selected_support), ('sb_vr_closure', finalize_sideband_plots__vr_closure),
                ('sb_release_audit', finalize_sideband_plots__release_audit), ('sb_comparison', finalize_sideband_plots__comparison),
                ('sb_nominal_sideband_data', finalize_sideband_plots__nominal_sideband_data),
                ('sb_meeting_conclusion', finalize_sideband_plots__meeting_conclusion)]
    return [(builder(result), name) for name, builder in builders]


def finalize_sideband_plots__nominal_data_mask(result):
    """Enforce the upstream nominal release decisions before displaying counts."""
    summary = finalize_sideband_plots__table(result, 'sbd_summary')
    required = {'candidate_id', 'display_allowed', 'reason', 'data_B', 'data_C',
                'data_D', 'pred', 'pred_error', 'nominal_A_evaluated'}
    if not required.issubset(summary):
        raise AssertionError('Nominal sideband DATA need explicit release and A-blinding records.')
    if summary.candidate_id.duplicated().any() or summary.display_allowed.isna().any():
        raise AssertionError('Nominal sideband DATA release records must be unique and explicit.')
    if summary.nominal_A_evaluated.isna().any() or summary.nominal_A_evaluated.astype(bool).any():
        raise AssertionError('The sideband display must never evaluate nominal A DATA.')
    fields = [name for name in summary if name.startswith(('data_', 'var_')) or
              name in {'pred', 'pred_error', 'pred_rel_error'}]
    allowed = summary.display_allowed.astype(bool)
    if not summary.loc[~allowed, fields].isna().all().all():
        raise AssertionError('Withheld nominal sideband DATA are not masked.')
    if not finalize_sideband_plots_np.isfinite(summary.loc[allowed, ['data_B', 'data_C', 'data_D']].to_numpy(float)).all():
        raise AssertionError('Every released nominal sideband count must be finite.')
    regions = finalize_sideband_plots__table(result, 'sbd_regions')
    if not regions.region.isin(list('BCD')).all():
        raise AssertionError('Only nominal B/C/D belong in the released sideband table.')
    if regions.display_allowed.isna().any():
        raise AssertionError('Every nominal regional release decision must be explicit.')
    fields = [name for name in ['value', 'variance', 'error', 'count'] if name in regions]
    if not regions.loc[~regions.display_allowed.astype(bool), fields].isna().all().all():
        raise AssertionError('Withheld nominal regional counts are not masked.')
    if set(regions.candidate_id) != set(summary.candidate_id):
        raise AssertionError('Nominal summary and regional DATA must describe the same candidates.')
    for _, row in summary.iterrows():
        regional = regions.loc[regions.candidate_id.eq(row.candidate_id)].set_index('region')
        if len(regional) != 3 or set(regional.index) != set('BCD'):
            raise AssertionError('Every nominal candidate must have exactly B, C and D release records.')
        if not regional.display_allowed.astype(bool).eq(bool(row.display_allowed)).all():
            raise AssertionError('Nominal summary and regional DATA release decisions disagree.')
        if row.display_allowed:
            finalize_sideband_plots_np.testing.assert_allclose(regional.loc[list('BCD'), 'value'].to_numpy(float),
                                       [row['data_'+r] for r in 'BCD'])
    return summary


def finalize_sideband_plots__nominal_sideband_data(result):
    summary = finalize_sideband_plots__nominal_data_mask(result).set_index('candidate_id')
    rows = finalize_sideband_plots__selected(result)
    data = summary.reindex(rows.candidate_id)
    if data.display_allowed.isna().any():
        raise AssertionError('Each shortlisted candidate needs a nominal sideband release decision.')
    fig, axes = finalize_sideband_plots__canvas('Nominal sidebands: approved DATA counts and uncorrected prediction',
                       cols=2, height=8.7, bottom=.25, top=.85)
    left, right = axes[0]
    left.axis('off')
    contents = []
    for _, row in rows.iterrows():
        record = data.loc[row.candidate_id]
        counts = [f'{record["data_"+r]:g}' for r in 'BCD'] if record.display_allowed else ['—']*3
        contents.append([finalize_sideband_plots__label(row)] + counts + ['SHOW' if record.display_allowed else 'HOLD'])
    table = left.table(cellText=contents, colLabels=['Definition', 'DATA B', 'DATA C', 'DATA D', 'Release'],
                       cellLoc='center', colWidths=[.30, .17, .17, .17, .19], bbox=[0, 0, 1, 1])
    finalize_sideband_plots__style_table(table, 9)
    left.set_title('New B/C/D definitions only; A is never evaluated', fontsize=12, pad=13)
    y = finalize_sideband_plots_np.arange(len(rows))
    values, errors = data.pred.to_numpy(float), data.pred_error.to_numpy(float)
    allowed = data.display_allowed.astype(bool).to_numpy()
    finite = allowed & finalize_sideband_plots_np.isfinite(values) & finalize_sideband_plots_np.isfinite(errors) & (errors >= 0)
    right.errorbar(values[finite], y[finite], xerr=errors[finite], fmt='o',
                   color='#0072B2', capsize=4, markersize=6)
    for i, (_, row) in enumerate(rows.iterrows()):
        record = data.loc[row.candidate_id]
        if finite[i]:
            precision = f' ({errors[i]/values[i]:.1%})' if values[i] > 0 else ''
            right.text(.98, i-.19, f'{values[i]:.2f} ± {errors[i]:.2f}' + precision,
                       transform=right.get_yaxis_transform(), fontsize=9, ha='right', color='#005087')
        else:
            reasons = {
                'BCD_CONTAMINATION_AT_OR_ABOVE_10_PERCENT': 'BCD contamination ≥10%',
                'PROTECTED_A_OVERLAP_MC_COUNTERFACTUAL_ONLY': 'Overlaps protected A; MC comparison only',
                'DATA_EVALUATION_DISABLED': 'Release eligible; DATA not evaluated',
                'UNSAFE_OR_UNSUPPORTED_RECONSTRUCTABLE_COMPLEMENT': 'Unsafe/unsupported reconstructed buffer',
            }
            label = 'Prediction undefined' if allowed[i] else reasons.get(str(record.reason), str(record.reason).replace('_', ' '))
            label = '\n'.join(finalize_sideband_plots_textwrap.wrap(label, width=58))
            right.text(.02, i, label, transform=right.get_yaxis_transform(),
                       fontsize=8, va='center', color='#777777')
    if finite.any():
        lower, upper = finalize_sideband_plots_np.nanmin(values[finite]-errors[finite]), finalize_sideband_plots_np.nanmax(values[finite]+errors[finite])
        span = max(float(upper-lower), float(abs(upper))*.1, 1.)
        right.set_xlim(min(0., lower)-.04*span, max(1., upper)+.12*span)
    else:
        right.set_xlim(0, 1)
    right.set(ylim=(len(rows)-.5, -.5), yticks=y,
              yticklabels=[finalize_sideband_plots__label(row) for _, row in rows.iterrows()],
              xlabel='Uncorrected DATA B×C/D prediction', title='Statistical propagation only')
    right.tick_params(axis='y', labelsize=8)
    right.axvline(0, color='#999999', lw=1)
    finalize_sideband_plots__caption(fig, 'Only independently approved B/C/D selections are read. Nominal A and the protected A union remain blind; withheld counts and predictions stay NaN.\n'
             'Release requires the all-120-model nominal BCD purity gate, protected-geometry safety and the complementary-disclosure audit.\n'
             'The displayed prediction uses B×C/D with first-order disjoint-count error propagation. It has no closure correction or associated systematic uncertainty.\n'
             'Sparse sidebands require a Poisson/count likelihood for inference. This is a statistics check, not a nominal DATA closure test or a DATA-constrained significance.')
    return fig


def finalize_sideband_plots__meeting_conclusion(result):
    """Summarize the requested leading definition using actual released tables."""
    chosen = finalize_sideband_plots__selected(result)
    definitions = []
    for strategy, iso, phi in [('gap', .3, 2.), ('loose', .3, 2.4)]:
        xf, yf = ('b_iso', 'c_phi') if strategy == 'gap' else ('a_iso', 'a_phi')
        selected = chosen.loc[chosen.strategy.eq(strategy) & finalize_sideband_plots_np.isclose(chosen[xf], iso) & finalize_sideband_plots_np.isclose(chosen[yf], phi)]
        if len(selected) != 1:
            raise AssertionError('The meeting proposal needs the requested leading gap and safe relaxed comparator.')
        definitions.append(selected.iloc[0])
    data = finalize_sideband_plots__nominal_data_mask(result).set_index('candidate_id')
    vr = finalize_sideband_plots__data_mask(finalize_sideband_plots__table(result, 'sbvr_closure')).set_index(['candidate_id', 'scope'])
    columns = []
    for row in definitions:
        nominal = data.loc[row.candidate_id]
        if not nominal.display_allowed:
            raise AssertionError('The meeting conclusion must not expose a withheld nominal sideband result.')
        closure = {}
        for scope in ['VR2_PP', 'VR2_PD', 'VR1_PP', 'VR1_PD']:
            key = (row.candidate_id, scope)
            if key in vr.index:
                record = vr.loc[key]
                closure[scope] = f'{record.kappa:.3f} ± {record.kappa_error:.3f}'
            else:
                closure[scope] = 'withheld'
        columns.append([
            f'I < {row.a_iso:g}; |Δφ| ≥ {row.a_phi:g}',
            f'{row.z_known_b_ratio_WP2_median:.1%}',
            f'{row.max_S_B_BCD:.2%}',
            ' / '.join(f'{nominal["data_"+r]:g}' for r in 'BCD'),
            f'{nominal.pred:.2f} ± {nominal.pred_error:.2f} ({nominal.pred_rel_error:.1%})',
            f'{row.neff_C:.2f}',
            closure['VR2_PP'] + ' / ' + closure['VR2_PD'],
            closure['VR1_PP'] + ' / ' + closure['VR1_PD'],
        ])
    quantities = ['A boundary', 'Median direct-MC Z / WP2', 'Maximum nominal BCD S/B',
                  'Nominal DATA B / C / D', 'Uncorrected DATA BC/D ± stat',
                  'Nominal C effective MC count', 'Intermediate DATA κ: PP / PD',
                  'Low-mass DATA κ: PP / PD']
    contents = [[name, columns[0][i], columns[1][i]] for i, name in enumerate(quantities)]
    fig, axes = finalize_sideband_plots__canvas('Meeting proposal: retain WP2 A and develop the buffered transfer',
                       cols=1, height=10.8, bottom=.46, top=.85)
    ax = axes[0, 0]
    ax.axis('off')
    gap, loose = definitions
    table = ax.table(cellText=contents, colLabels=['Evidence',
                       f'Leading buffer: I_H={gap.b_iso:g}, φ_L={gap.c_phi:g}',
                       f'Relaxed comparator: I={loose.a_iso:g}, φ={loose.a_phi:g}'],
                     colWidths=[.34, .33, .33], cellLoc='center', bbox=[0, 0, 1, 1])
    finalize_sideband_plots__style_table(table, 10)
    fig.text(.075, .89, 'A proposal for transfer and uncertainty modeling; the nominal DD background model and final limits are not yet validated.',
             fontsize=10.5, color='#25354A')
    fig.text(.075, .40, 'Why lead with the fixed-A buffer', fontsize=12, weight='bold', color='#9F4200')
    fig.text(.075, .371,
             'Preserves WP2 A and its direct-MC sensitivity.\n'
             'Approved B/C/D contain usable observed counts.\n'
             'Intermediate PP/PD provide direct closure checks.\n'
             'Sparse C MC and low-mass nonclosure remain limiting.',
             fontsize=11, va='top', linespacing=1.5, color='#25354A')
    fig.text(.55, .40, 'Why retain the relaxed comparator', fontsize=12, weight='bold', color='#005087')
    fig.text(.55, .371,
             'Provides more margin below the contamination threshold.\n'
             'Gives a more precise raw DATA sideband prediction.\n'
             'Expanding A reduces the direct-MC sensitivity.\n'
             'Mass-dependent closure remains unresolved here too.',
             fontsize=11, va='top', linespacing=1.5, color='#25354A')
    margin = 100*(.1-gap.max_S_B_BCD)
    fig.text(.075, .215, 'Before fixing the background model', fontsize=12, weight='bold', color='#17263B')
    fig.text(.075, .185,
             'Quantify the mass/displacement transfer and assign a closure uncertainty; use a count likelihood for sparse sidebands.\n'
             f'The leading buffer is only {margin:.2f} percentage points below the 10% gate, so the central-MC purity pass is fragile.\n'
             'The largest isolation gap remains a statistics stress test, not the leading choice.',
             fontsize=11, va='top', linespacing=1.5, color='#25354A')
    finalize_sideband_plots__caption(fig, 'Values are read from the nominal MC scan and independently approved nominal/VR DATA tables. Nominal A remains blind.\n'
             'Z compares the same matching signal models with direct MC background; it is not a DATA-constrained significance or an expected limit.\n'
             'Raw BC/D includes statistical propagation only. VR agreement does not by itself establish a transferable nominal DD correction; low-mass closure shows residual mass dependence.')
    return fig


# ========================================================================
# abcd_iso_buffer_plots: original study API
# ========================================================================
"""Capped isolation scans and explicitly masked, source-separated ABCD figures.

Only validated calculation products are consumed. No original DATA are read,
no gate is changed and no new DATA disclosure is made by this module.
"""
import numpy as finalize_iso_buffer_plots_np
import pandas as finalize_iso_buffer_plots_pd
import matplotlib.pyplot as finalize_iso_buffer_plots_plt
import matplotlib.patheffects as finalize_iso_buffer_plots_patheffects
from matplotlib.colors import LogNorm as finalize_iso_buffer_plots_LogNorm; from matplotlib.colors import Normalize as finalize_iso_buffer_plots_Normalize
from matplotlib.lines import Line2D as finalize_iso_buffer_plots_Line2D
from matplotlib.patches import Patch as finalize_iso_buffer_plots_Patch; from matplotlib.patches import Rectangle as finalize_iso_buffer_plots_Rectangle

finalize_iso_buffer_plots__canvas = _FINALIZE_NAMESPACES['abcd_final_check_plots']._canvas; finalize_iso_buffer_plots__caption = _FINALIZE_NAMESPACES['abcd_final_check_plots']._caption; finalize_iso_buffer_plots__style_table = _FINALIZE_NAMESPACES['abcd_final_check_plots']._style_table
finalize_iso_buffer_plots__data_mask = _FINALIZE_NAMESPACES['abcd_sideband_plots']._data_mask

finalize_iso_buffer_plots_ANGLES = (2.4, 2.6, 2.7, 2.8)
finalize_iso_buffer_plots_COLORS = {2.4:'#9467BD', 2.6:'#0072B2', 2.7:'#D55E00', 2.8:'#009E73'}
finalize_iso_buffer_plots_MASS_TITLES = {'low':'Low mass: 0 ≤ MJJ < 150 GeV',
               'intermediate':'Intermediate mass: 150 ≤ MJJ < 300 GeV',
               'all':'Mass inclusive: MJJ ≥ 0 GeV'}
finalize_iso_buffer_plots_SCOPE_PREFIX = {'low':'VR1', 'intermediate':'VR2', 'all':'INCLUSIVE'}


def finalize_iso_buffer_plots__scan(result):
    rows = result['tables']['if_scan'].copy()
    required = {'candidate_id','a_iso','a_phi','max_S_B_BCD','purity_pass',
        'z_known_b_median','z_mc_stat_diagnostic_median','s_sqrt_s_plus_b_median'}
    if not required.issubset(rows) or len(rows)!=20 or rows.candidate_id.duplicated().any():
        raise AssertionError('Require 20 unique capped nominal MC comparisons.')
    if set(finalize_iso_buffer_plots_np.round(rows.a_phi,10))!=set(finalize_iso_buffer_plots_ANGLES) or not rows.groupby('a_phi').size().eq(5).all():
        raise AssertionError('Require the native angles 2.4, 2.6, 2.7 and 2.8, with five isolation cuts each.')
    if not finalize_iso_buffer_plots_np.allclose(sorted(rows.a_iso.unique()), [.2,.25,.3,.4,.5]):
        raise AssertionError('Isolation boundaries must lie on the requested grid through 0.5.')
    if not rows.signal_hypotheses.eq(120).all():
        raise AssertionError('The contamination envelope requires all 120 hypotheses.')
    return rows


def finalize_iso_buffer_plots__best(result, role='global_best'):
    selected=result['tables']['if_selection']
    selected=selected.loc[selected.role.eq(role)]
    if len(selected)!=1:
        raise AssertionError('Exactly one explicit selection is required: '+role)
    return finalize_iso_buffer_plots__scan(result).set_index('candidate_id',drop=False).loc[selected.iloc[0].candidate_id]


def finalize_iso_buffer_plots__final(result):
    return finalize_iso_buffer_plots__best(result,'global_operational_best')


def finalize_iso_buffer_plots__iso_axis(ax, values, label='Isolation boundary I_A (larger = looser)'):
    values=finalize_iso_buffer_plots_np.asarray(sorted(set(values)),float)
    ax.set_xticks(values)
    ax.set_xticklabels([f'{value:g}' for value in values],fontsize=8)
    ax.set_xlim(values.min()-.018,values.max()+.018)
    ax.set_xlabel(label)


def finalize_iso_buffer_plots__closure_axis(ax):
    ax.axhspan(.9,1.1,color='#85B879',alpha=.18,zorder=0)
    ax.axhline(1,color='#5F775B',ls='--',lw=1.1)
    ax.set(ylim=(0,2),ylabel='Closure κ = A / (B×C/D)')


def finalize_iso_buffer_plots__closure_points(ax,x,rows,color,marker='o',label=None,level=0,connect=True):
    x=finalize_iso_buffer_plots_np.asarray(x,float)
    y=rows.kappa.to_numpy(float); error=rows.kappa_error.to_numpy(float)
    valid=finalize_iso_buffer_plots_np.isfinite(y)&finalize_iso_buffer_plots_np.isfinite(error)&(error>=0)
    ax.errorbar(x,finalize_iso_buffer_plots_np.where(valid,y,finalize_iso_buffer_plots_np.nan),yerr=finalize_iso_buffer_plots_np.where(valid,error,finalize_iso_buffer_plots_np.nan),
        fmt=marker,ls='-' if connect else 'none',color=color,ms=4,lw=1,capsize=2,label=label)
    for index in finalize_iso_buffer_plots_np.flatnonzero(valid&((y<0)|(y>2))):
        upper=y[index]>2
        edge=1.94-.025*level if upper else .06+.025*level
        ax.plot(x[index],edge,'^' if upper else 'v',color=color,ms=5)
        ax.text(x[index],1.72-.10*level if upper else .20+.10*level,
            f'{y[index]:.2g}',ha='center',fontsize=6.5,color=color,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.65,pad=.2))
    return int(finalize_iso_buffer_plots_np.sum(valid&((y-error<0)|(y+error>2))))


def finalize_iso_buffer_plots__tradeoff(result):
    rows,selected=finalize_iso_buffer_plots__scan(result),finalize_iso_buffer_plots__final(result)
    fig,axes=finalize_iso_buffer_plots__canvas('Relax isolation: absolute sensitivity versus sideband contamination',cols=3,height=7.8,bottom=.30,top=.82)
    metric='s_sqrt_s_plus_b_median'
    for angle in finalize_iso_buffer_plots_ANGLES:
        part=rows.loc[finalize_iso_buffer_plots_np.isclose(rows.a_phi,angle)].sort_values('a_iso')
        color=finalize_iso_buffer_plots_COLORS[angle]
        axes[0,0].plot(part.a_iso,part[metric],'-o',color=color,ms=5,label=f'|Δφ| boundary {angle:g}')
        axes[0,1].plot(part.a_iso,part.z_known_b_median,'-o',color=color,ms=5)
        axes[0,1].plot(part.a_iso,part.z_mc_stat_diagnostic_median,'--',color=color,alpha=.65)
        axes[0,2].plot(part.a_iso,100*part.max_S_B_BCD,'-o',color=color,ms=5)
        passed=part.purity_pass.astype(bool)
        axes[0,2].scatter(part.loc[passed,'a_iso'],100*part.loc[passed,'max_S_B_BCD'],
            s=110,facecolors='none',edgecolors='#2D8B45',lw=1.6,zorder=4)
    for ax,value in zip(axes[0],[selected[metric],selected.z_known_b_median,100*selected.max_S_B_BCD]):
        ax.scatter([selected.a_iso],[value],marker='*',s=190,color='#B32424',edgecolors='white',zorder=6)
    axes[0,0].set(title='S / √(S+B)',ylabel='Absolute median S / √(S+B)')
    axes[0,1].set(title='Asimov significance Z',ylabel='Absolute median Z')
    axes[0,2].set(title='Worst nominal B/C/D contamination',ylabel='Maximum S/B_MC over 120 hypotheses [%]',yscale='log')
    axes[0,2].axhline(10,color='#A33333',ls='--',lw=1.3)
    for ax in axes[0]: finalize_iso_buffer_plots__iso_axis(ax,rows.a_iso)
    handles=[finalize_iso_buffer_plots_Line2D([],[],color=finalize_iso_buffer_plots_COLORS[a],label=f'|Δφ|={a:g}') for a in finalize_iso_buffer_plots_ANGLES]
    handles += [finalize_iso_buffer_plots_Line2D([],[],color='#444444',ls='--',label='Dashed Z: finite-MC diagnostic'),
        finalize_iso_buffer_plots_Line2D([],[],marker='*',color='#B32424',ls='none',ms=10,label='Selected loose optimum'),
        finalize_iso_buffer_plots_Line2D([],[],marker='o',mfc='none',mec='#2D8B45',ls='none',ms=8,label='BCD MC contamination <10%')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.135),ncol=4,frameon=False,fontsize=8.5)
    finalize_iso_buffer_plots__caption(fig,'The active grid contains I=0.20,0.25,0.30,0.40,0.50 and native |Δφ|=2.4,2.6,2.7,2.8; no 2.5 boundary is available in these histograms.\n'
        'Both sensitivity panels show absolute per-hypothesis metrics, then take the median over the same matched 60 signal models. No normalization to a reference working point is applied.\n'
        'The star maximizes absolute median known-background Z among points passing B/C/D MC contamination <10% for all 120 hypotheses. Historical A unions do not veto the ranking.\n'
        'A-based sensitivity does not include the full ABCD likelihood, control statistics or transfer uncertainty. Neighbouring cuts share events.')
    return fig


def finalize_iso_buffer_plots__nominal_closure(result):
    rows=finalize_iso_buffer_plots__scan(result)
    fig,axes=finalize_iso_buffer_plots__canvas('Isolation relaxation: nominal MC closure and control precision',cols=2,height=8.6,bottom=.25,top=.84)
    clipped=0
    for level,angle in enumerate(finalize_iso_buffer_plots_ANGLES):
        part=rows.loc[finalize_iso_buffer_plots_np.isclose(rows.a_phi,angle)].sort_values('a_iso')
        clipped+=finalize_iso_buffer_plots__closure_points(axes[0,0],part.a_iso,part,finalize_iso_buffer_plots_COLORS[angle],label=f'|Δφ|={angle:g}',level=level)
        relative=finalize_iso_buffer_plots_np.divide(part.pred_error,part.pred,out=finalize_iso_buffer_plots_np.full(len(part),finalize_iso_buffer_plots_np.nan),where=part.pred>0)
        axes[0,1].plot(part.a_iso,relative,'o-',color=finalize_iso_buffer_plots_COLORS[angle],ms=4,label=f'|Δφ|={angle:g}')
    finalize_iso_buffer_plots__closure_axis(axes[0,0]); axes[0,0].set_title('Nominal MC: direct A versus B×C/D')
    axes[0,1].set(title='Nominal MC sideband precision',ylabel='σ(B×C/D) / (B×C/D)',yscale='log')
    for ax in axes[0]: finalize_iso_buffer_plots__iso_axis(ax,rows.a_iso); ax.legend(frameon=False,fontsize=9)
    finalize_iso_buffer_plots__caption(fig,f'Nominal background MC only. The green band 0.9≤κ≤1.1 is a ±10% visual guide, not an uncertainty band or a confidence interval.\n'
        f'All κ panels use 0–2. Off-frame central values have directional markers and numeric labels; {clipped} nominal-MC error ranges extend outside the frame. Full values remain in the tables.\n'
        'Uncertainties propagate disjoint-region sumw² at first order; sparse weighted-MC ratios need not have Gaussian coverage. Curves share events.\n'
        'No nominal DATA A or nominal DATA closure is evaluated.')
    return fig


def finalize_iso_buffer_plots__selection(result):
    rows,chosen=finalize_iso_buffer_plots__scan(result),finalize_iso_buffer_plots__final(result)
    absolute_finite=rows.loc[rows.purity_pass].sort_values('z_mc_stat_diagnostic_median',ascending=False).iloc[0]
    fig,axes=finalize_iso_buffer_plots__canvas('Selected loose optimum: absolute sensitivity after B/C/D purity',cols=1,height=7.8,bottom=.25,top=.85)
    ax=axes[0,0]; ax.axis('off'); contents=[]
    for angle in finalize_iso_buffer_plots_ANGLES:
        passing=rows.loc[finalize_iso_buffer_plots_np.isclose(rows.a_phi,angle)&rows.purity_pass].sort_values('a_iso')
        if passing.empty:
            contents.append([f'{angle:g}','No pass ≤0.5','—','—','—','—','—','—'])
        else:
            best=passing.sort_values(['z_known_b_median','a_iso'],ascending=[False,True]).iloc[0]
            contents.append([f'{angle:g}',f'{passing.iloc[0].a_iso:g}',f'{best.a_iso:g}',
                f'{100*best.max_S_B_BCD:.3f}%',f'{best.s_sqrt_s_plus_b_median:.4f}',f'{best.z_known_b_median:.4f}',
                f'{best.z_mc_stat_diagnostic_median:.4f}','SELECTED' if best.candidate_id==chosen.candidate_id else 'Angle optimum'])
    table=ax.table(cellText=contents,colLabels=['Fixed |Δφ|','First passing iso','Best iso\nat this angle',
        'Maximum BCD S/B','Median S/√(S+B)','Median known-B Z','Median finite-MC Z','Final choice'],cellLoc='center',
        colWidths=[.08,.12,.10,.14,.15,.15,.12,.14],bbox=[0,.30,1,.64]);finalize_iso_buffer_plots__style_table(table,9.3)
    fig.text(.075,.33,f'Selected loose boundary: ({chosen.a_iso:g},{chosen.a_phi:g}), absolute median known-Z={chosen.z_known_b_median:.4f}.\n'
        f'Maximum B/C/D contamination = {100*chosen.max_S_B_BCD:.3f}% over all 120 signal hypotheses; the selected nominal A remains blinded.',
        fontsize=11,weight='bold',color='#25354A')
    fig.text(.075,.21,f'The absolute finite-MC diagnostic also peaks at ({absolute_finite.a_iso:g},{absolute_finite.a_phi:g}) among contamination-passing points.\n'
        'Nominal MC selects the boundary; observed control counts and VR closure validate the selected definition separately.',
        fontsize=10.5,color='#8D3429')
    finalize_iso_buffer_plots__caption(fig,'All three sensitivity columns are absolute medians over the same matched 60 hypotheses; contamination uses all 120 separate hypotheses.\n'
        'Selection uses absolute median known-background Z after central MC contamination <10%; historical candidate A unions are not selection constraints.\n'
        'MC statistical uncertainty, VR closure and transfer uncertainty still enter the final statistical interpretation. Nominal A remains blinded.')
    return fig


def finalize_iso_buffer_plots__vr_scan(result,mass,source):
    rows,chosen=finalize_iso_buffer_plots__scan(result),finalize_iso_buffer_plots__final(result)
    frame=result['tables']['if_vr_closure'];finalize_iso_buffer_plots__data_mask(frame)
    states=finalize_iso_buffer_plots__visible_states(frame,mass,chosen.candidate_id)
    fig,axes=finalize_iso_buffer_plots__canvas(f'Isolation scan · {source} VR closure · '+finalize_iso_buffer_plots_MASS_TITLES[mass],cols=len(states),height=8.8,bottom=.28,top=.81)
    fig.text(.075,.865,f'Selected loose optimum: I<{chosen.a_iso:g}, |Δφ|≥{chosen.a_phi:g}. Only {source} closure is shown.',fontsize=11,color='#25354A')
    for ax,state in zip(axes[0],states):
        scope=finalize_iso_buffer_plots_SCOPE_PREFIX[mass]+'_'+state;clipped=0
        for level,angle in enumerate(finalize_iso_buffer_plots_ANGLES):
            part=rows.loc[finalize_iso_buffer_plots_np.isclose(rows.a_phi,angle)].sort_values('a_iso')
            values=frame.loc[frame.scope.eq(scope)&frame.source.eq(source)].set_index('candidate_id').reindex(part.candidate_id)
            if values.source.isna().any():raise AssertionError('Every scan point requires an explicit source/mask row.')
            clipped+=finalize_iso_buffer_plots__closure_points(ax,part.a_iso,values,finalize_iso_buffer_plots_COLORS[angle],level=level)
            if finalize_iso_buffer_plots_np.isclose(angle,chosen.a_phi):
                point=values.loc[chosen.candidate_id]
                finalize_iso_buffer_plots__selected_note(ax,point,source)
                if source=='DATA':
                    missing=~values.display_allowed.to_numpy(bool)
                    ax.scatter(part.a_iso.to_numpy()[missing],finalize_iso_buffer_plots_np.full(missing.sum(),-.02),marker='x',color='#777777',s=22,transform=ax.get_xaxis_transform(),clip_on=False)
                if finalize_iso_buffer_plots_np.isfinite(point.kappa):
                    finalize_iso_buffer_plots__closure_points(ax,[chosen.a_iso],finalize_iso_buffer_plots_pd.DataFrame([point]),'#B32424',marker='D',connect=False)
        finalize_iso_buffer_plots__closure_axis(ax);finalize_iso_buffer_plots__iso_axis(ax,rows.a_iso)
        ax.axvline(chosen.a_iso,color='#B32424',ls=':',lw=1)
        ax.set_title(state);ax.tick_params(axis='x',pad=10)
        if clipped:ax.text(.025,.025,f'{clipped} error ranges leave frame',transform=ax.transAxes,fontsize=6.5,color='#666666')
    handles=[finalize_iso_buffer_plots_Line2D([],[],color=finalize_iso_buffer_plots_COLORS[a],label=f'|Δφ|={a:g}') for a in finalize_iso_buffer_plots_ANGLES]
    handles += [finalize_iso_buffer_plots_Line2D([],[],color='#B32424',marker='D',ls='none',label='Selected loose optimum'),finalize_iso_buffer_plots_Patch(facecolor='#85B879',alpha=.25,label='±10% visual guide')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.135),ncol=3,frameon=False,fontsize=9)
    finalize_iso_buffer_plots__caption(fig,f'{source} only: '+', '.join(states)+' are explicit, disjoint displacement categories. Green 0.9–1.1 is a visual reference, not a confidence interval.\n'
        'The κ axis is fixed at 0–2; arrows and labels preserve off-frame central values, and full errors remain in the exported numerical tables.\n'
        'DATA curves contain only matching previously released integrals. A missing/withheld value is not zero; neighbouring cuts are correlated.\n'
        'The red comparison is selected using nominal MC sensitivity and B/C/D contamination. Low-mass DD is included only when its selected DATA closure is approved; nominal A remains blinded.')
    return fig


def finalize_iso_buffer_plots__selected_note(ax,point,source):
    allowed=source=='MC' or bool(point.display_allowed)
    if allowed:
        text=f'Selected {source} κ={point.kappa:.3f}±{point.kappa_error:.3f}' if finalize_iso_buffer_plots_np.isfinite(point.kappa) else f'Selected {source} κ undefined'
        text+='\nA/B/C/D='+ '/'.join(f'{point[r]:.3g}' for r in 'ABCD')
    else:text='Selected DATA withheld\n'+finalize_iso_buffer_plots__reason(point.reason)
    ax.text(.025,.97,text,transform=ax.transAxes,va='top',fontsize=7.7,bbox=dict(facecolor='white',edgecolor='none',alpha=.9))


def finalize_iso_buffer_plots__gap_nominal_closure(result):
    scan=result['tables']['sideband_scan']
    rows=scan.loc[scan.strategy.eq('gap')&(scan.b_iso<=.5+1e-10)]
    geometry=finalize_iso_buffer_plots__geometry(result,'gap')
    chosen=rows.loc[rows.candidate_id.eq(geometry['candidate_id'])]
    if len(chosen)!=1:raise AssertionError('Require the selected buffer nominal MC closure.')
    fig,axes=finalize_iso_buffer_plots__canvas('Buffer sideband starts: nominal MC closure with A fixed',cols=2,height=8.6,bottom=.26,top=.84)
    specifications=[('b_iso','c_phi',[1.5,2.,2.4]),('c_phi','b_iso',[.2,geometry['b_iso'],.5])]
    for ax,(axis,fixed,values) in zip(axes[0],specifications):
        for level,(color,value) in enumerate(zip(['#0072B2','#D55E00','#009E73'],values)):
            part=rows.loc[finalize_iso_buffer_plots_np.isclose(rows[fixed],value)].sort_values(axis)
            if len(part):finalize_iso_buffer_plots__closure_points(ax,part[axis],part,color,level=level,label=f'{"φ_L" if fixed=="c_phi" else "I_H"}={value:g}')
        finalize_iso_buffer_plots__closure_points(ax,chosen[axis],chosen,'#B32424',marker='D',connect=False,label='Selected buffer')
        ax.axvline(geometry[axis],color='#B32424',ls=':',lw=1)
        finalize_iso_buffer_plots__closure_axis(ax);ax.legend(frameon=False,fontsize=9)
        x=sorted(rows[axis].unique());ax.set_xticks(x);ax.tick_params(axis='x',labelsize=8)
        ax.set_xlabel('B/D isolation start I_H (≤0.5)' if axis=='b_iso' else 'C/D angle upper edge φ_L [rad]')
        ax.set_title('Move B/D away in isolation' if axis=='b_iso' else 'Move C/D away in angle')
    finalize_iso_buffer_plots__caption(fig,f'A is fixed at I<0.20 and |Δφ|≥2.7 in both panels. Red marks the selected I_H={geometry["b_iso"]:.2f}, φ_L={geometry["c_phi"]:g}; these axes move the sideband starts, not the signal-region boundary.\n'
        'The angular sideband edge φ_L can be below 2.4 even though the ordinary A-boundary scan stays in 2.4–2.8. Every isolation sideband start shown is ≤0.5.\n'
        'Nominal MC only. All κ axes use 0–2 and the green 0.9–1.1 region is a ±10% visual guide, not a confidence interval. Off-frame central values are labelled.\n'
        'First-order MC ratio errors and correlated neighbouring cuts are diagnostics. No nominal DATA A or nominal DATA closure is evaluated.')
    return fig


def finalize_iso_buffer_plots__path_frame(result,study,mass,source):
    frame=result['tables']['vd_closure1d']
    axis='a_iso' if study=='loose' else 'c_phi'
    rows=frame.loc[frame.study.eq(study)&frame.mass_slice.eq(mass)&frame.source.eq(source)&frame.scan_axis.eq(axis)].copy()
    if rows.empty:raise AssertionError('Missing final comparison path: '+study+'/'+mass+'/'+source)
    if (rows.a_iso>.5+1e-10).any() or (rows.b_iso>.5+1e-10).any():
        raise AssertionError('Final comparison paths must respect the isolation cap.')
    finalize_iso_buffer_plots__data_mask(frame)
    return rows.sort_values('scan_value'),axis


def finalize_iso_buffer_plots__gap_vr_scan(result,mass,source):
    rows,axis=finalize_iso_buffer_plots__path_frame(result,'gap',mass,source)
    geometry=finalize_iso_buffer_plots__geometry(result,'gap')
    all_rows=result['tables']['vd_closure1d']
    states=finalize_iso_buffer_plots__visible_states(all_rows.loc[all_rows.study.eq('gap')],mass,geometry['candidate_id'])
    fig,axes=finalize_iso_buffer_plots__canvas(f'Buffer sideband scan · {source} VR closure · '+finalize_iso_buffer_plots_MASS_TITLES[mass],cols=len(states),height=8.6,bottom=.26,top=.83)
    for ax,state in zip(axes[0],states):
        part=rows.loc[rows.plane.eq(state)].sort_values('scan_value')
        point=part.loc[part.selected_point.astype(bool)]
        if len(point)!=1:raise AssertionError('Require one selected buffer point per VR/source.')
        finalize_iso_buffer_plots__closure_points(ax,part.scan_value,part,'#D55E00' if source=='MC' else '#222222')
        finalize_iso_buffer_plots__closure_points(ax,point.scan_value,point,'#B32424',marker='D',connect=False)
        finalize_iso_buffer_plots__selected_note(ax,point.iloc[0],source)
        if source=='DATA':
            missing=~part.display_allowed.to_numpy(bool)
            ax.scatter(part.scan_value.to_numpy()[missing],finalize_iso_buffer_plots_np.full(missing.sum(),-.02),marker='x',s=22,color='#777777',transform=ax.get_xaxis_transform(),clip_on=False)
        finalize_iso_buffer_plots__closure_axis(ax);ax.axvline(float(point.iloc[0].scan_value),color='#B32424',ls=':',lw=1)
        ax.set(title=state,xlabel='C/D angle upper edge φ_L [rad]',xticks=part.scan_value)
        ax.set_xlim(float(part.scan_value.min())-.07,float(part.scan_value.max())+.07)
        ax.tick_params(axis='x',labelsize=8,pad=10,labelrotation=60)
    finalize_iso_buffer_plots__caption(fig,f'{source} only. A stays at I<{geometry["a_iso"]:.2f} and |Δφ|≥{geometry["a_phi"]:g}; B/D start at I_H={geometry["b_iso"]:.2f}. Only the C/D angle upper edge is scanned, with φ_L={geometry["c_phi"]:g} selected.\n'
        'The scan moves the buffer edge, not the A boundary. κ=1 is closure; green 0.9–1.1 is a ±10% visual guide, not a confidence interval.\n'
        'κ axes are fixed at 0–2, with off-frame central values labelled and full errors retained in tables. A withheld DATA point is not zero.\n'
        'Integrated DATA permission is evaluated separately from already-public 2D-bin display permission. Nominal DATA A remains blinded.')
    return fig


def finalize_iso_buffer_plots__final_1d(result,study,mass):
    rows,axis=finalize_iso_buffer_plots__path_frame(result,study,mass,'DATA')
    label='Selected loose optimum' if study=='loose' else 'Fixed WP2 A + selected buffer'
    selected=rows.loc[rows.selected_point.astype(bool)].candidate_id.unique()
    if len(selected)!=1:raise AssertionError('A final comparison path must select one common working point.')
    states=finalize_iso_buffer_plots__visible_states(rows,mass,selected[0])
    fig,axes=finalize_iso_buffer_plots__canvas(f'{label} · final DATA closure · '+finalize_iso_buffer_plots_MASS_TITLES[mass],rows=3,cols=len(states),height=10.8,bottom=.25,top=.88)
    fig.subplots_adjust(hspace=.27)
    for j,state in enumerate(states):
        part=rows.loc[rows.plane.eq(state)].sort_values('scan_value')
        chosen=part.loc[part.selected_point.astype(bool)]
        if len(chosen)!=1:raise AssertionError('Require one final path point per displacement state.')
        x=part.scan_value.to_numpy(float);point=chosen.iloc[0]
        observed=part.A.to_numpy(float);prediction=part.pred.to_numpy(float)
        axes[0,j].errorbar(x,observed,yerr=part.A_error,fmt='o-',color='#222222',ms=4,capsize=2,label='Observed DATA A')
        axes[0,j].errorbar(x,prediction,yerr=part.pred_error,fmt='s--',color='#0072B2',ms=4,capsize=2,label='DATA B×C/D')
        axes[0,j].set(title=state,ylabel='DATA events',ylim=(0,None))
        if not finalize_iso_buffer_plots_np.isfinite(observed).any() and not finalize_iso_buffer_plots_np.isfinite(prediction).any():
            axes[0,j].set(ylim=(0,1),yticks=[])
            axes[0,j].text(.5,.5,'Integrated DATA withheld',transform=axes[0,j].transAxes,
                           ha='center',va='center',fontsize=10,color='#777777')
        if point.display_allowed:
            text=f'Selected A={point.A:.2f}±{point.A_error:.2f}\nB×C/D={point.pred:.2f}±{point.pred_error:.2f}\nκ={point.kappa:.3f}±{point.kappa_error:.3f}'
        else:text='Selected integrated DATA withheld\n'+finalize_iso_buffer_plots__reason(point.reason)
        axes[0,j].text(.025,.96,text,transform=axes[0,j].transAxes,va='top',fontsize=7.5,bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
        finalize_iso_buffer_plots__closure_points(axes[1,j],x,part,'#222222')
        finalize_iso_buffer_plots__closure_points(axes[1,j],chosen.scan_value,chosen,'#B32424',marker='D',connect=False)
        finalize_iso_buffer_plots__closure_axis(axes[1,j])
        axes[2,j].plot(x,100*part.max_S_B,'o-',color='#9467BD',ms=4)
        axes[2,j].axhline(10,color='#A33333',ls='--',lw=1)
        axes[2,j].set(ylabel='Max regional S/B_MC [%]',yscale='log')
        for ax in axes[:,j]:
            ax.axvline(point.scan_value,color='#B32424',ls=':',lw=1)
            ax.set_xticks(x);ax.tick_params(axis='x',labelsize=7.5,labelrotation=60 if study=='gap' else 0)
            margin=.018 if axis=='a_iso' else .07
            ax.set_xlim(x.min()-margin,x.max()+margin)
        xlabel=f'Isolation boundary I_A; φ_A={point.a_phi:g}' if study=='loose' else f'C/D angle upper edge φ_L; I_H={point.b_iso:.2f}'
        axes[2,j].set_xlabel(xlabel,fontsize=9)
        for ax in axes[:2,j]:
            missing=~part.display_allowed.to_numpy(bool)
            ax.scatter(x[missing],finalize_iso_buffer_plots_np.full(missing.sum(),-.025),marker='x',s=20,color='#777777',transform=ax.get_xaxis_transform(),clip_on=False)
    handles=[finalize_iso_buffer_plots_Line2D([],[],color='#222222',marker='o',label='Observed DATA A'),
        finalize_iso_buffer_plots_Line2D([],[],color='#0072B2',marker='s',ls='--',label='DATA sideband prediction B×C/D'),
        finalize_iso_buffer_plots_Line2D([],[],color='#B32424',ls=':',label='Selected final comparison'),
        finalize_iso_buffer_plots_Patch(facecolor='#85B879',alpha=.25,label='κ ±10% visual guide')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.125),ncol=2,frameon=False,fontsize=9)
    finalize_iso_buffer_plots__caption(fig,'The selected boundaries are marked in red, with neighbouring native cuts retained as stability checks. Only explicitly approved DATA integrals appear in the first two rows.\n'
        'Observed A and predicted B×C/D use disjoint DATA counts with their existing propagated uncertainties. κ=1 indicates closure; 0.9–1.1 is a visual guide, not a confidence interval.\n'
        'The bottom row is MC contamination: the maximum over all A/B/C/D regions and 120 separate signal hypotheses. Passing 10% is necessary but does not override disclosure protection.\n'
        'Neighbouring cuts are correlated. A missing DATA value is not zero; partial visible 2D bins are never substituted for an integrated DATA yield. Nominal A remains blinded.')
    return fig


def finalize_iso_buffer_plots__planes(result,study='gap'):
    planes=result.get('selected_planes',{}).get(study)
    if planes is None and study=='gap':planes=result.get('buffer_planes',result.get('planes',{}))
    if not planes:raise AssertionError('Missing selected distribution planes: '+study)
    required=['Nominal_DD']+[p+'_'+s for p in ('VR1','VR2','INCLUSIVE') for s in ('PP','PD','DP')]+['VR1_DD']
    if not set(required).issubset(planes):raise AssertionError('Require nominal MC and ten explicit VR scopes.')
    for scope in required:
        plane=planes[scope];shape=(len(plane['xedges'])-1,len(plane['yedges'])-1)
        for key in ('MC','DATA','variance_MC','variance_DATA','approved','display_allowed','max_S_B',
                    'complementary_pass','release_provenance','joint_release_pass'):
            if finalize_iso_buffer_plots_np.asarray(plane[key]).shape!=shape:raise AssertionError(scope+': incompatible display shape')
        background=finalize_iso_buffer_plots_np.asarray(plane['MC'],float);data=finalize_iso_buffer_plots_np.asarray(plane['DATA'],float);variance=finalize_iso_buffer_plots_np.asarray(plane['variance_DATA'],float)
        shown=finalize_iso_buffer_plots_np.asarray(plane['display_allowed'],bool);approved=finalize_iso_buffer_plots_np.asarray(plane['approved'],bool)
        complementary=finalize_iso_buffer_plots_np.asarray(plane['complementary_pass'],bool)
        provenance=finalize_iso_buffer_plots_np.asarray(plane['release_provenance'],str)
        previous=provenance=='PREVIOUS_PUBLIC_BIN'
        new_joint=provenance=='NEW_JOINT_AUDITED_DP_BIN'
        if finalize_iso_buffer_plots_np.any(approved&~complementary):raise AssertionError('Every eligible display bin needs an explicit complementary-disclosure pass.')
        if finalize_iso_buffer_plots_np.any(approved&~(previous|new_joint)):raise AssertionError('An eligible display bin needs an explicit public or jointly audited origin.')
        if new_joint.any():
            if scope!='INCLUSIVE_DP':raise AssertionError('New display-bin disclosure is restricted to audited mass-inclusive DP.')
            if not finalize_iso_buffer_plots_np.asarray(plane['joint_release_pass'],bool)[new_joint].all():raise AssertionError('Every new DP bin needs the joint release proof.')
        if not finalize_iso_buffer_plots_np.isfinite(background).all() or finalize_iso_buffer_plots_np.any(background<0):raise AssertionError('MC bins must be finite and nonnegative.')
        if finalize_iso_buffer_plots_np.any(shown&~approved):raise AssertionError('An ineligible DATA bin cannot be displayed.')
        if not finalize_iso_buffer_plots_np.isnan(data[~shown]).all() or not finalize_iso_buffer_plots_np.isnan(variance[~shown]).all():raise AssertionError('Withheld DATA bins and variances must be NaN.')
        if scope=='Nominal_DD' and (shown.any() or finalize_iso_buffer_plots_np.isfinite(data).any()):raise AssertionError('Nominal DATA must never enter a distribution.')
        if shown.any():
            ratios=finalize_iso_buffer_plots_np.asarray(plane['max_S_B'],float)[shown]
            if not (finalize_iso_buffer_plots_np.isfinite(ratios)&(ratios>=0)&(ratios<.1)).all() or not (background[shown]>1e-8).all():raise AssertionError('Visible DATA bins need positive MC support and all-model purity <10%.')
            if not finalize_iso_buffer_plots_np.isfinite(data[shown]).all() or finalize_iso_buffer_plots_np.any(data[shown]<0) or not finalize_iso_buffer_plots_np.array_equal(data[shown],variance[shown]) or not finalize_iso_buffer_plots_np.array_equal(data[shown],finalize_iso_buffer_plots_np.rint(data[shown])):raise AssertionError('Visible DATA must be nonnegative integer counts with variance equal to count.')
    return planes


def finalize_iso_buffer_plots__geometry(result,study):
    if study=='gap':
        selected=result['selections']['gap']
        return {key:selected[key] for key in ('candidate_id','a_iso','a_phi','b_iso','c_phi')}
    row=finalize_iso_buffer_plots__final(result)
    return {key:row[key] for key in ('candidate_id','a_iso','a_phi','b_iso','c_phi')}


def finalize_iso_buffer_plots__cut(ax,value,vertical,color,style='-',width=2.2):
    line=ax.axvline(value,color=color,ls=style,lw=width,zorder=7) if vertical else ax.axhline(value,color=color,ls=style,lw=width,zorder=7)
    line.set_path_effects([finalize_iso_buffer_plots_patheffects.Stroke(linewidth=width+2.7,foreground='white',alpha=.95),finalize_iso_buffer_plots_patheffects.Normal()])
    return line


def finalize_iso_buffer_plots__overlay(ax,xmax,geometry):
    ai,ap,bi,cp=[float(geometry[key]) for key in ('a_iso','a_phi','b_iso','c_phi')]
    gap=(bi>ai+1e-9 or cp<ap-1e-9)
    if bi>ai+1e-9:ax.add_patch(finalize_iso_buffer_plots_Rectangle((ai,0),bi-ai,finalize_iso_buffer_plots_np.pi,facecolor='#FFC857',alpha=.17,edgecolor='#A56D00',hatch='///',lw=.6,zorder=4))
    if cp<ap-1e-9:ax.add_patch(finalize_iso_buffer_plots_Rectangle((0,cp),xmax,ap-cp,facecolor='#FFC857',alpha=.13,edgecolor='#A56D00',hatch='///',lw=.6,zorder=4))
    for value in sorted(set((ai,bi))):finalize_iso_buffer_plots__cut(ax,value,True,'#B32035')
    for value in sorted(set((ap,cp))):finalize_iso_buffer_plots__cut(ax,value,False,'#B32035')
    for label,x,y in [('A',ai/2,(ap+finalize_iso_buffer_plots_np.pi)/2),('B',(bi+xmax)/2,(ap+finalize_iso_buffer_plots_np.pi)/2),('C',ai/2,cp/2),('D',(bi+xmax)/2,cp/2)]:
        ax.text(x,y,label,ha='center',va='center',fontsize=15,weight='bold',color='#17263B',bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1.2),zorder=9)
    ax.set(xlim=(0,xmax),ylim=(0,finalize_iso_buffer_plots_np.pi),xlabel='Muon-LJ isolation',ylabel='|Δφ| [rad]')
    ticks=sorted(set([0,1,cp,ap,finalize_iso_buffer_plots_np.pi]));ax.set_yticks(ticks,labels=['π' if finalize_iso_buffer_plots_np.isclose(v,finalize_iso_buffer_plots_np.pi) else f'{v:g}' for v in ticks]);ax.grid(False)
    return gap


def finalize_iso_buffer_plots__plane_image(ax,plane,source,norm,geometry=None,xmax=2.,title=''):
    values=finalize_iso_buffer_plots_np.asarray(plane[source],float);xx=finalize_iso_buffer_plots_np.asarray(plane['xedges']);yy=finalize_iso_buffer_plots_np.asarray(plane['yedges'])
    cmap=finalize_iso_buffer_plots_plt.get_cmap('viridis').copy();cmap.set_bad('white')
    array=finalize_iso_buffer_plots_np.ma.masked_where(~finalize_iso_buffer_plots_np.isfinite(values)|(values<=0),values)
    mesh=ax.pcolormesh(xx,yy,array.T,norm=norm,cmap=cmap,shading='flat',rasterized=True)
    if source=='DATA':
        for ix,iy in finalize_iso_buffer_plots_np.argwhere(~finalize_iso_buffer_plots_np.asarray(plane['display_allowed'],bool)):
            ax.add_patch(finalize_iso_buffer_plots_Rectangle((xx[ix],yy[iy]),xx[ix+1]-xx[ix],yy[iy+1]-yy[iy],facecolor='#CFD4DC',edgecolor='#939DAA',lw=.25,hatch='xx',zorder=3))
    if geometry is not None:finalize_iso_buffer_plots__overlay(ax,xmax,geometry)
    else:ax.set(xlim=(0,xmax),ylim=(0,finalize_iso_buffer_plots_np.pi),xlabel='Muon-LJ isolation',ylabel='|Δφ| [rad]');ax.grid(False)
    ax.set_title(title,fontsize=12)
    if source=='DATA':
        shown=finalize_iso_buffer_plots_np.asarray(plane['display_allowed'],bool)
        gate=bool(plane.get('closure_gate',plane['integrated_allowed']))
        note=f'DATA: {shown.sum()}/{shown.size} display bins shown\nIntegrated closure: '+('allowed' if gate else 'withheld')
        new_joint=shown&(finalize_iso_buffer_plots_np.asarray(plane['release_provenance'],str)=='NEW_JOINT_AUDITED_DP_BIN')
        if new_joint.any():note+=f'\n{new_joint.sum()} bins passed the joint disclosure audit'
        ax.text(.02,.02,note,transform=ax.transAxes,va='bottom',ha='left',fontsize=7.1,bbox=dict(facecolor='white',edgecolor='none',alpha=.92),zorder=10)
    return mesh


def finalize_iso_buffer_plots__distribution_legend(fig,data=True,gap=True,bottom=.135):
    handles=[finalize_iso_buffer_plots_Line2D([],[],color='#B32035',lw=2.2,label='ABCD boundary (white halo)'),finalize_iso_buffer_plots_Patch(facecolor='white',edgecolor='#999999',label='Zero in a displayed bin')]
    if gap:handles.insert(0,finalize_iso_buffer_plots_Patch(facecolor='#FFF0C2',edgecolor='#A56D00',hatch='///',label='Buffer excluded from A/B/C/D'))
    if data:handles.append(finalize_iso_buffer_plots_Patch(facecolor='#CFD4DC',edgecolor='#939DAA',hatch='xx',label='DATA blinded / unsupported'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,bottom),ncol=2 if len(handles)>3 else 3,frameon=False,fontsize=9)


def finalize_iso_buffer_plots__nominal_plane(result):
    plane=finalize_iso_buffer_plots__planes(result,'gap')['Nominal_DD'];chosen=finalize_iso_buffer_plots__final(result)
    fig,axes=finalize_iso_buffer_plots__canvas('Nominal background MC: native bins and visible boundary choices',cols=2,height=9.,bottom=.28,top=.83)
    fig.subplots_adjust(right=.93,wspace=.35);norm,_=finalize_iso_buffer_plots__norm([plane['MC']])
    definitions=[(.3,2.6,'#0072B2','--','Reference (0.30, 2.6)'),(.2,2.7,'#D55E00','-','WP2 A (0.20, 2.7)'),(chosen.a_iso,chosen.a_phi,'#C02672',':',f'Selected loose ({chosen.a_iso:g}, {chosen.a_phi:g})')]
    for ax,xmax,title in zip(axes[0],(2.,.6),('Full display: isolation 0–2','Near-boundary view: isolation 0–0.6')):
        mesh=finalize_iso_buffer_plots__plane_image(ax,plane,'MC',norm,None,xmax,title)
        for iso,phi,color,style,label in definitions:
            finalize_iso_buffer_plots__cut(ax,iso,True,color,style,2.6);finalize_iso_buffer_plots__cut(ax,phi,False,color,style,2.6)
        ax.set_yticks([0,1,2,2.4,2.6,2.7,finalize_iso_buffer_plots_np.pi],labels=['0','1','2','2.4','2.6','2.7','π'])
        fig.colorbar(mesh,ax=ax,fraction=.046,pad=.025,label='Background MC events / display bin')
    fig.legend(handles=[finalize_iso_buffer_plots_Line2D([],[],color=color,ls=style,lw=2.6,label=label) for _,_,color,style,label in definitions],loc='lower center',bbox_to_anchor=(.5,.145),ncol=3,frameon=False,fontsize=9)
    finalize_iso_buffer_plots__caption(fig,'Nominal DD, MJJ≥150 GeV, partner isolation in [0,0.5). Native MC bin contents are redrawn directly; thick coloured boundaries with white halos remain visible over the heatmap.\n'
        f'Each ordinary A is left of its isolation boundary and above its angle boundary. The selected loose boundary is I={chosen.a_iso:g}, |Δφ|={chosen.a_phi:g}.\n'
        'Both panels share the same logarithmic colour scale. Counts are per displayed bin, not a density: the final [1,2] isolation display bin contains all I≥1 tails/overflow; angular flows are folded.\n'
        'The right panel changes only the view. Nominal DATA A and the full nominal DATA plane are never evaluated.')
    return fig


def finalize_iso_buffer_plots__nominal_distribution(result):
    plane=finalize_iso_buffer_plots__planes(result,'gap')['Nominal_DD'];geometry=finalize_iso_buffer_plots__geometry(result,'gap')
    fig,axes=finalize_iso_buffer_plots__canvas('Fixed-A buffer in nominal background MC',cols=2,height=8.9,bottom=.27,top=.83)
    fig.subplots_adjust(right=.93,wspace=.35);norm,_=finalize_iso_buffer_plots__norm([plane['MC']])
    for ax,xmax,title in zip(axes[0],(2.,.6),('Full display: isolation 0–2','Near-boundary view: isolation 0–0.6')):
        mesh=finalize_iso_buffer_plots__plane_image(ax,plane,'MC',norm,geometry,xmax,title)
        fig.colorbar(mesh,ax=ax,fraction=.046,pad=.025,label='Background MC events / display bin')
    finalize_iso_buffer_plots__distribution_legend(fig,data=False,gap=True)
    finalize_iso_buffer_plots__caption(fig,f'Nominal MC only. A: I<{geometry["a_iso"]:.2f}, φ≥{geometry["a_phi"]:g}; B: I≥{geometry["b_iso"]:.2f}, φ≥{geometry["a_phi"]:g}; C: I<{geometry["a_iso"]:.2f}, φ<{geometry["c_phi"]:g}; D: I≥{geometry["b_iso"]:.2f}, φ<{geometry["c_phi"]:g}. Yellow hatching marks unused buffer strips.\n'
        'The two panels use identical native counts and a shared logarithmic colour scale; the zoom changes only the view. ABCD integrals retain their full support.\n'
        'Counts are per display bin, not a density. The final [1,2] isolation bin includes all I≥1 tails/overflow; angular flows enter the first/last bins; isolation underflow is excluded.\n'
        'Boundary lines use white halos for visibility. Nominal DATA are neither selected nor plotted.')
    return fig


def finalize_iso_buffer_plots__vr_distributions(result,study,mass,source):
    planes=finalize_iso_buffer_plots__planes(result,study);geometry=finalize_iso_buffer_plots__geometry(result,study);states=['PP','PD','DP']
    if mass=='low' and bool(planes['VR1_DD'].get('closure_gate',planes['VR1_DD']['integrated_allowed'])):
        states=['DD',*states]
    scopes=[finalize_iso_buffer_plots_SCOPE_PREFIX[mass]+'_'+s for s in states]
    label='Selected loose optimum' if study=='loose' else 'Fixed-A buffer'
    fig,axes=finalize_iso_buffer_plots__canvas(f'{label} · {source} 2D · '+finalize_iso_buffer_plots_MASS_TITLES[mass],cols=len(states),height=8.9,bottom=.29,top=.83)
    fig.subplots_adjust(right=.93,wspace=.39);norm,exists=finalize_iso_buffer_plots__norm([planes[s][source] for s in scopes])
    for ax,state,scope in zip(axes[0],states,scopes):
        plane=planes[scope];mesh=finalize_iso_buffer_plots__plane_image(ax,plane,source,norm,geometry,title=state)
        bar=fig.colorbar(mesh,ax=ax,fraction=.040,pad=.020,label='MC weighted events / bin' if source=='MC' else 'DATA events / bin')
        values=finalize_iso_buffer_plots_np.asarray(plane[source],float)
        if not exists or not finalize_iso_buffer_plots_np.any(finalize_iso_buffer_plots_np.isfinite(values)&(values>0)):bar.ax.set_visible(False)
    if len(states)==4:
        for ax in axes[0,1:]:ax.set_ylabel('')
    finalize_iso_buffer_plots__distribution_legend(fig,data=source=='DATA',gap=study=='gap',bottom=.15)
    finalize_iso_buffer_plots__caption(fig,f'{source} only. The same selected '+('ordinary loose boundaries' if study=='loose' else 'WP2 A and buffered sidebands')+' are drawn in '+', '.join(states)+'. Colour is events per displayed bin, with one shared source-specific logarithmic scale.\n'
        'Grey cross-hatching denotes withheld DATA, not zero. Visible zeros are white. Every displayed bin passes all-120-model S/B_MC<10% and its explicit disclosure audit.\n'
        'DATA are either exact reuses of public bins or individually joint-audited inclusive-DP bins. Integrated closure remains separately gated; no visible-bin subtotal replaces a regional yield.\n'
        'The last isolation display bin [1,2] includes all I≥1 tails/overflow; angular flows are folded into edge bins; isolation underflow is excluded. Nominal A remains blinded.')
    return fig
def finalize_iso_buffer_plots__reason(reason):
    text = str(reason)
    if 'CONTAMINATION' in text or 'INVALID_BACKGROUND' in text:
        return 'purity / MC support'
    if 'COMPLEMENT' in text or 'DISCLOSURE' in text:
        return 'complement protection'
    if 'PREVIOUS' in text or 'RELEASE' in text or 'PUBLIC' in text:
        return 'not previously released'
    return text.replace('_', ' ').lower()[:48]



def finalize_iso_buffer_plots__norm(arrays):
    positive = [finalize_iso_buffer_plots_np.asarray(a, float)[finalize_iso_buffer_plots_np.isfinite(a) & (finalize_iso_buffer_plots_np.asarray(a)>0)] for a in arrays]
    values = finalize_iso_buffer_plots_np.concatenate(positive) if positive else finalize_iso_buffer_plots_np.array([])
    if len(values) == 0:
        return finalize_iso_buffer_plots_Normalize(0, 1), False
    low, high = values.min(), values.max()
    if finalize_iso_buffer_plots_np.isclose(low, high):
        low, high = low*.9, high*1.1
    return finalize_iso_buffer_plots_LogNorm(vmin=low, vmax=high), True



def finalize_iso_buffer_plots__nominal_fraction_table(result):
    frame = result['tables']['if_nominal_data_contamination'].copy()
    if 'focused_definition' in frame and 'previous_nominal_candidate_id' in frame:
        focused = frame.loc[frame.focused_definition.astype(bool)]
        aliases = set(focused.previous_nominal_candidate_id.dropna())
        frame = frame.loc[~frame.candidate_id.isin(aliases)].copy()
    if set(frame.region) != set('BCD') or frame.duplicated(['candidate_id', 'region']).any():
        raise AssertionError('The nominal DATA-fraction diagnostic is restricted to unique B/C/D rows.')
    if frame.nominal_A_evaluated.any():
        raise AssertionError('The fractions must never evaluate nominal DATA A.')
    if not frame.signal_hypotheses.eq(120).all():
        raise AssertionError('The nominal contamination comparison requires 120 separate hypotheses.')
    data_fields = ['data_count', 'data_count_error', 'max_S_over_N_data',
                   'max_S_over_N_data_error', 'max_S_over_N_minus_S']
    if not frame.loc[~frame.data_available.astype(bool), data_fields].isna().all().all():
        raise AssertionError('Unavailable nominal DATA denominators and ratios must remain NaN.')
    if (frame.data_available.astype(bool) != frame.display_allowed.astype(bool)).any():
        raise AssertionError('Every available nominal DATA diagnostic needs an explicit display decision.')
    return frame



def finalize_iso_buffer_plots__definition_label(row):
    if row.strategy == 'gap':
        return f'Buffer I_H={row.b_iso:g}, φ_L={row.c_phi:g}'
    return f'Iso<{row.a_iso:g}, |Δφ|≥{row.a_phi:g}'



def finalize_iso_buffer_plots__nominal_data_contamination(result):
    frame = finalize_iso_buffer_plots__nominal_fraction_table(result)
    ids = [finalize_iso_buffer_plots__final(result).candidate_id, finalize_iso_buffer_plots__geometry(result,'gap')['candidate_id']]
    if not set(ids).issubset(set(frame.candidate_id)):
        raise AssertionError('Both final approaches need explicit nominal contamination rows.')
    palette = ['#0072B2', '#D55E00', '#777777', '#009E73', '#B32424', '#CC79A7']
    fig, axes = finalize_iso_buffer_plots__canvas('Nominal B/C/D: MC contamination and control counts',
                       cols=1, height=7.4, bottom=.46, top=.84)
    fig.set_size_inches(13.5, 7.4)
    ax = axes[0, 0]
    offsets = finalize_iso_buffer_plots_np.linspace(-.12, .12, len(ids))
    positive = []
    contents, handles = [], []
    for index, (name, shift) in enumerate(zip(ids, offsets)):
        part = frame.loc[frame.candidate_id.eq(name)].set_index('region').loc[list('BCD')]
        row, color = part.iloc[0], palette[index % len(palette)]
        x = finalize_iso_buffer_plots_np.arange(3)+shift
        mc = 100*part.max_S_B_MC.to_numpy(float)
        ax.plot(x, mc, 's', color=color, mfc='white', ms=6)
        visible = part.data_available.to_numpy(bool)
        positive.extend(mc[finalize_iso_buffer_plots_np.isfinite(mc) & (mc > 0)])
        data_label = ('New B/C/D check' if part.new_data_evaluated.any() else 'Reused controls') if visible.all() else 'DATA withheld'
        contents.append([finalize_iso_buffer_plots__definition_label(row),
                         *[f'{value:.0f}' if finalize_iso_buffer_plots_np.isfinite(value) else '—' for value in part.data_count],
                         f'{100*part.max_S_B_MC.max():.3f}%', data_label])
        handles.append(finalize_iso_buffer_plots_Line2D([], [], color=color, marker='s', mfc='white', ls='none', label=finalize_iso_buffer_plots__definition_label(row)))
    ax.set(title='Central MC contamination gate', ylabel='Maximum S/B_MC over 120 hypotheses [%]',
           yscale='log', xticks=finalize_iso_buffer_plots_np.arange(3),
           xticklabels=['B · high isolation', 'C · low |Δφ|', 'D · both sidebands'],
           ylim=(min(positive)*.55, max(20., max(positive)*1.7)))
    ax.axhline(10, color='#A33333', ls='--', lw=1.2)
    ax.tick_params(axis='x', labelsize=9)
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .365), ncol=2, frameon=False, fontsize=9)
    table_ax = fig.add_axes([.075, .13, .895, .19])
    table_ax.axis('off')
    table = table_ax.table(cellText=contents,
        colLabels=['Definition', 'DATA B', 'DATA C', 'DATA D', 'Max BCD S/B_MC', 'Nominal DATA status'],
        cellLoc='center', colWidths=[.29, .085, .085, .085, .18, .275], bbox=[0, 0, 1, 1])
    finalize_iso_buffer_plots__style_table(table, 9.2)
    finalize_iso_buffer_plots__caption(fig, 'MC contamination is the maximum S/B_MC over all 120 separate signal hypotheses in each control region; the dashed line is the central-value 10% gate.\n'
             'The table gives the observed nominal B/C/D counts separately. Different signal hypotheses are never summed.\n'
             'Both final definitions were fixed from MC before the control-count queries. Nominal DATA A remains blinded.')
    return fig



def finalize_iso_buffer_plots__approaches(result):
    selected = finalize_iso_buffer_plots__final(result)
    gap = finalize_iso_buffer_plots__geometry(result,'gap')
    fig, axes = finalize_iso_buffer_plots__canvas('Two approaches: loosen isolation and angle, or keep A and separate the sidebands',
                       cols=2, height=9., bottom=.27, top=.84)
    for ax in axes[0]:
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Muon-LJ isolation', ylabel='|Δφ|',
               xticks=[], yticks=[])
        ax.grid(False)
    left, right = axes[0]
    x, y = .55, .52
    for name, xx, yy, width, height, color in [
            ('A', 0, y, x, 1-y, '#FBE5E5'), ('B', x, y, 1-x, 1-y, '#E8EFF8'),
            ('C', 0, 0, x, y, '#E8EFF8'), ('D', x, 0, 1-x, y, '#E8EFF8')]:
        left.add_patch(finalize_iso_buffer_plots_Rectangle((xx, yy), width, height, facecolor=color, edgecolor='#8795A7'))
        tx, ty = (.42, .64) if name == 'A' else (xx+width/2, yy+height/2)
        left.text(tx, ty, name, ha='center', va='center', fontsize=25, weight='bold')
    left.add_patch(finalize_iso_buffer_plots_Rectangle((0, .70), .30, .30, fill=False, edgecolor='#D55E00', ls='--', lw=2))
    left.text(.15, .81, 'Original WP2 A', ha='center', fontsize=9, color='#9F4200')
    left.annotate('', xy=(.53, .94), xytext=(.31, .94),
                  arrowprops=dict(arrowstyle='->', lw=2, color='#9F2929'))
    left.annotate('', xy=(.15, .54), xytext=(.15, .69),
                  arrowprops=dict(arrowstyle='->', lw=2, color='#9F2929'))
    left.set_xticks([.30, x], ['0.20 (WP2)', f'{selected.a_iso:.2f} (loose)'], fontsize=9)
    left.set_yticks([y, .70], [f'{selected.a_phi:g} (loose)', '2.7 (WP2)'], fontsize=9)
    left.set_title('Loosen both boundaries: A expands right and down', fontsize=13)
    left.text(.5, -.13, f'Selected loose WP: I_A 0.20 → {selected.a_iso:.2f}; |Δφ| 2.7 → {selected.a_phi:g}.\n'
              'B/C/D share both updated boundaries.',
              transform=left.transAxes, ha='center', va='top', fontsize=10)
    right.set_facecolor('#eeeeee')
    xlo, xhi, ylo, yhi = .30, .55, .40, .70
    for name, xx, yy, width, height, color in [
            ('A', 0, yhi, xlo, 1-yhi, '#FBE5E5'), ('B', xhi, yhi, 1-xhi, 1-yhi, '#E8EFF8'),
            ('C', 0, 0, xlo, ylo, '#E8EFF8'), ('D', xhi, 0, 1-xhi, ylo, '#E8EFF8')]:
        right.add_patch(finalize_iso_buffer_plots_Rectangle((xx, yy), width, height, facecolor=color, edgecolor='#8795A7'))
        right.text(xx+width/2, yy+height/2, name, ha='center', va='center', fontsize=25, weight='bold')
    right.text(.15, .76, 'WP2 unchanged', fontsize=8.5, ha='center', color='#9F2929')
    right.text(.425, .55, 'buffer', ha='center', va='center', fontsize=13, color='#666666', rotation=90)
    right.text(.77, .55, 'buffer', ha='center', va='center', fontsize=13, color='#666666')
    right.set_title('Buffer: WP2 A fixed; B/C/D move away', fontsize=13)
    right.text(.5, -.09, f'A: I<{gap["a_iso"]:.2f}, φ≥{gap["a_phi"]:g}; B: I≥{gap["b_iso"]:.2f}, φ≥{gap["a_phi"]:g}\n'
               f'C: I<{gap["a_iso"]:.2f}, φ<{gap["c_phi"]:g}; D: I≥{gap["b_iso"]:.2f}, φ<{gap["c_phi"]:g}',
               transform=right.transAxes, ha='center', va='top', fontsize=10)
    finalize_iso_buffer_plots__caption(fig, 'Conceptual diagram; widths and heights are not proportional to the physical numerical boundaries. Left: increase the isolation cut and lower the angle cut; A expands in both directions.\n'
             'The capped scan repeats the comparison at native |Δφ|=2.4, 2.6, 2.7 and 2.8, with isolation ≤0.5. Right: keep the complete WP2 A definition and discard the isolation and angular buffer strips.\n'
             'Every candidate needs contamination and VR checks, with the final nominal A blinded. Retaining the same MC A sensitivity does not guarantee the same complete statistical-model sensitivity.\n'
             'Nominal DATA A remains blinded. This illustration introduces no DATA selection or new working-point authorization.')
    return fig



def finalize_iso_buffer_plots__meeting_comparison(result):
    iso=finalize_iso_buffer_plots__final(result)
    wp2=finalize_iso_buffer_plots__scan(result).set_index('candidate_id').loc['I_i0p2_p2p7']
    gap_id=finalize_iso_buffer_plots__geometry(result,'gap')['candidate_id']
    gap=result['tables']['sideband_scan'].set_index('candidate_id').loc[gap_id]
    fractions=finalize_iso_buffer_plots__nominal_fraction_table(result)
    ifrac=fractions.loc[fractions.candidate_id.eq(iso.candidate_id)].set_index('region')
    gfrac=fractions.loc[fractions.candidate_id.eq(gap_id)].set_index('region')
    summary=result['tables']['sbd_summary'].set_index('candidate_id')
    source_ids=[ifrac.iloc[0].get('previous_nominal_candidate_id'),iso.source_candidate_id,iso.candidate_id]
    source_ids=[identity for identity in source_ids if isinstance(identity,str) and identity in summary.index]
    if not source_ids:raise AssertionError('The selected loose point requires its exact nominal B/C/D summary.')
    idata,gdata=summary.loc[source_ids[0]],summary.loc[gap_id]
    def closure(study,state,mass='intermediate'):
        rows,axis=finalize_iso_buffer_plots__path_frame(result,study,mass,'DATA')
        row=rows.loc[rows.plane.eq(state)&rows.selected_point.astype(bool)]
        if len(row)!=1 or not bool(row.iloc[0].display_allowed):return 'DATA withheld'
        point=row.iloc[0]
        return f'{point.kappa:.3f} ± {point.kappa_error:.3f}'
    values=[['A definition',f'I<{iso.a_iso:g}, |Δφ|≥{iso.a_phi:g}',f'I<{gap.a_iso:.2f}, |Δφ|≥{gap.a_phi:g}'],
        ['Sideband boundaries',f'Ordinary I={iso.a_iso:g}, φ={iso.a_phi:g}',f'B/D: I≥{gap.b_iso:.2f}; C/D: φ<{gap.c_phi:g}'],
        ['Absolute median S/√(S+B)',f'{iso.s_sqrt_s_plus_b_median:.4f}',f'{wp2.s_sqrt_s_plus_b_median:.4f}'],
        ['Absolute median known-background Z',f'{iso.z_known_b_median:.4f}',f'{wp2.z_known_b_median:.4f}'],
        ['Absolute median finite-MC Z',f'{iso.z_mc_stat_diagnostic_median:.4f}',f'{wp2.z_mc_stat_diagnostic_median:.4f}'],
        ['Maximum nominal BCD S/B_MC',f'{100*iso.max_S_B_BCD:.3f}%',f'{100*gap.max_S_B_BCD:.3f}%'],
        ['Nominal DATA B / C / D',' / '.join(f'{ifrac.loc[r,"data_count"]:.0f}' for r in 'BCD'),' / '.join(f'{gfrac.loc[r,"data_count"]:.0f}' for r in 'BCD')],
        ['Uncorrected nominal B×C/D ± stat',f'{idata.pred:.2f} ± {idata.pred_error:.2f}',f'{gdata.pred:.2f} ± {gdata.pred_error:.2f}'],
        ['Intermediate PP DATA κ',closure('loose','PP'),closure('gap','PP')],
        ['Intermediate PD DATA κ',closure('loose','PD'),closure('gap','PD')],
        ['Intermediate DP DATA κ',closure('loose','DP'),closure('gap','DP')],
        ['Mass-inclusive PP DATA κ',closure('loose','PP','all'),closure('gap','PP','all')],
        ['Mass-inclusive PD DATA κ',closure('loose','PD','all'),closure('gap','PD','all')]]
    fig,axes=finalize_iso_buffer_plots__canvas('Final comparison: selected loose boundary versus fixed-A buffer',cols=1,height=8.4,bottom=.26,top=.87)
    ax=axes[0,0];ax.axis('off')
    table=ax.table(cellText=values,colLabels=['Evidence','Selected loose optimum','Fixed WP2 A + buffer'],
        cellLoc='center',colWidths=[.40,.30,.30],bbox=[0,.08,1,.90]);finalize_iso_buffer_plots__style_table(table,9.6)
    fig.text(.075,.215,f'The selected loose optimum is ({iso.a_iso:g},{iso.a_phi:g}); historical A unions no longer restrict MC optimization.\n'
        'The buffer preserves tight A and its direct-MC sensitivity; both selected definitions are checked with nominal B/C/D DATA.\n'
        'The full statistical interpretation includes sideband leakage, control-count uncertainty and the VR-derived transfer uncertainty.',
        fontsize=10.2,color='#25354A',va='top')
    finalize_iso_buffer_plots__caption(fig,'Sensitivity entries are absolute medians over the same matched 60 signal hypotheses. The buffer uses the identical WP2 A, hence its absolute MC-A metrics are those of WP2.\n'
        'Contamination uses direct background MC; observed control counts are listed separately. Raw B×C/D errors are statistical only; closure/transfer uncertainties remain separate.\n'
        'A-based significance omits the complete control-count and closure likelihood. Both final nominal A definitions remain blinded.')
    return fig


def finalize_iso_buffer_plots_iso_buffer_figure_definitions():
    rows=[('if_nominal_plane','Native nominal MC and boundary choices','Show reference, tight WP2 and selected loose cuts with high-contrast lines.'),
        ('if_approaches','Loosen isolation and angle or separate controls from A','Show both boundary shifts to the selected loose WP; keep the buffer A fixed.'),
        ('if_tradeoff','Capped sensitivity and contamination tradeoff','Compare absolute median S/sqrt(S+B), significance and BCD contamination.'),
        ('if_nominal_closure','Loose nominal MC closure','Separate nominal MC closure from VR tests.'),
        ('if_selection','Capped candidate selection','Select the absolute known-Z maximum after B/C/D purity, including angles without a pass.')]
    for source in ('mc','data'):
        for mass in ('low','intermediate','all'):
            rows.append((f'if_vr_{source}_{mass}',f'Loose {source.upper()} closure: '+finalize_iso_buffer_plots_MASS_TITLES[mass],'Source-separated isolation scan with a 0–2 closure axis.'))
    rows.append(('gap_nominal_closure','Buffer nominal MC closure','Scan sideband starts while A remains fixed.'))
    for source in ('mc','data'):
        for mass in ('low','intermediate','all'):
            rows.append((f'gap_vr_{source}_{mass}',f'Buffer {source.upper()} closure: '+finalize_iso_buffer_plots_MASS_TITLES[mass],'Scan the angular buffer edge with an explicit selected point.'))
    rows.append(('buffer2d_nominal','Nominal MC with a fixed-A buffer','Show full and zoomed native distributions with visible boundaries.'))
    for prefix in ('loose2d','buffer2d'):
        for source in ('mc','data'):
            for mass in ('low','intermediate','all'):
                rows.append((f'{prefix}_{source}_{mass}',f'{prefix} {source.upper()}: '+finalize_iso_buffer_plots_MASS_TITLES[mass],'Keep source and integrated/bin display permissions separate.'))
    for prefix in ('loose','buffer'):
        for mass in ('low','intermediate','all'):
            rows.append((f'{prefix}_final_1d_{mass}',f'{prefix.title()} final DATA closure: '+finalize_iso_buffer_plots_MASS_TITLES[mass],'Show observed A, BCD prediction, closure and MC contamination around the selected cut.'))
    rows.extend([('if_nominal_data_contamination','Nominal MC contamination and control counts','Show MC B/C/D contamination and observed nominal control counts separately.'),
        ('if_meeting_comparison','Selected loose versus fixed-A buffer','State the final candidate definitions, control diagnostics and transfer-model limitations.')])
    return finalize_iso_buffer_plots_pd.DataFrame(rows,columns=['name','title','purpose'])


def finalize_iso_buffer_plots_make_iso_buffer_figures(result,config):
    """Return 39 explicitly source-separated figures without accessing DATA."""
    if config['topology']!='2mu2e':raise ValueError('This helper is restricted to 2mu2e.')
    finalize_iso_buffer_plots__scan(result);finalize_iso_buffer_plots__data_mask(result['tables']['if_vr_closure'])
    finalize_iso_buffer_plots__planes(result,'loose');finalize_iso_buffer_plots__planes(result,'gap')
    figures=[(finalize_iso_buffer_plots__nominal_plane(result),'if_nominal_plane'),(finalize_iso_buffer_plots__approaches(result),'if_approaches'),
        (finalize_iso_buffer_plots__tradeoff(result),'if_tradeoff'),(finalize_iso_buffer_plots__nominal_closure(result),'if_nominal_closure'),(finalize_iso_buffer_plots__selection(result),'if_selection')]
    for source in ('MC','DATA'):
        for mass in ('low','intermediate','all'):
            figures.append((finalize_iso_buffer_plots__vr_scan(result,mass,source),f'if_vr_{source.lower()}_{mass}'))
    figures.append((finalize_iso_buffer_plots__gap_nominal_closure(result),'gap_nominal_closure'))
    for source in ('MC','DATA'):
        for mass in ('low','intermediate','all'):
            figures.append((finalize_iso_buffer_plots__gap_vr_scan(result,mass,source),f'gap_vr_{source.lower()}_{mass}'))
    figures.append((finalize_iso_buffer_plots__nominal_distribution(result),'buffer2d_nominal'))
    for study,prefix in [('loose','loose2d'),('gap','buffer2d')]:
        for source in ('MC','DATA'):
            for mass in ('low','intermediate','all'):
                figures.append((finalize_iso_buffer_plots__vr_distributions(result,study,mass,source),f'{prefix}_{source.lower()}_{mass}'))
    for study,prefix in [('loose','loose'),('gap','buffer')]:
        for mass in ('low','intermediate','all'):
            figures.append((finalize_iso_buffer_plots__final_1d(result,study,mass),f'{prefix}_final_1d_{mass}'))
    figures.extend([(finalize_iso_buffer_plots__nominal_data_contamination(result),'if_nominal_data_contamination'),(finalize_iso_buffer_plots__meeting_comparison(result),'if_meeting_comparison')])
    if [name for _,name in figures]!=finalize_iso_buffer_plots_iso_buffer_figure_definitions().name.tolist():raise AssertionError('Figure manifest mismatch.')
    return figures


def finalize_iso_buffer_plots__visible_states(frame,mass,candidate_id):
    """Include low-mass DD only when its selected integrated DATA is released."""
    states=['PP','PD','DP']
    if mass!='low':return states
    selected=frame.loc[frame.source.eq('DATA')&frame.scope.eq('VR1_DD')&frame.candidate_id.eq(candidate_id)]
    if selected.empty:return states
    allowed=selected.display_allowed.astype(bool)
    if allowed.nunique()!=1:raise AssertionError('Duplicate selected DD rows disagree on integrated release permission.')
    return ['DD',*states] if bool(allowed.iloc[0]) else states


# ========================================================================
# abcd_final_comparison_plots: original study API
# ========================================================================
"""Direct final-approach comparisons from validated MC and public VR tables.

This module does not access original DATA or change any selection or mask.
The numerical comparison tables retain the exact hypothesis and scan mappings.
"""
import re as finalize_final_comparison_plots_re

import numpy as finalize_final_comparison_plots_np
import pandas as finalize_final_comparison_plots_pd
from matplotlib.lines import Line2D as finalize_final_comparison_plots_Line2D
from matplotlib.patches import Patch as finalize_final_comparison_plots_Patch

finalize_final_comparison_plots__canvas = _FINALIZE_NAMESPACES['abcd_final_check_plots']._canvas; finalize_final_comparison_plots__caption = _FINALIZE_NAMESPACES['abcd_final_check_plots']._caption
finalize_final_comparison_plots__closure_axis = _FINALIZE_NAMESPACES['abcd_iso_buffer_plots']._closure_axis; finalize_final_comparison_plots__closure_points = _FINALIZE_NAMESPACES['abcd_iso_buffer_plots']._closure_points


finalize_final_comparison_plots_COLORS = {'loose': '#0072B2', 'gap': '#D55E00'}
finalize_final_comparison_plots_LABELS = {'loose': 'Loose boundary', 'gap': 'Fixed-A buffer'}
finalize_final_comparison_plots_METRICS = ('s_sqrt_s_plus_b', 'z_known_b', 'z_mc_stat_diagnostic')


def finalize_final_comparison_plots_final_comparison_definitions():
    return finalize_final_comparison_plots_pd.DataFrame([
        ('final_intermediate_overlay', 'Intermediate PP/PD/DP: loose and buffer overlay',
         'Compare DATA A, BCD predictions, closure and maximum MC contamination on a common isolation-edge scan.'),
        ('final_significance_comparison', 'Absolute significance for the two final approaches',
         'Compare all 60 identical matching signal hypotheses using absolute S/sqrt(S+B), known-background Z and finite-MC Z.')],
        columns=['name', 'title', 'purpose'])


def finalize_final_comparison_plots_make_final_comparison_figures(result):
    """Return two final comparisons without opening DATA or recomputing yields."""
    tables = finalize_final_comparison_plots_final_comparison_tables(result)
    return [(finalize_final_comparison_plots__intermediate_overlay(tables['final_comparison_vr'], result['selections']),
             'final_intermediate_overlay'),
            (finalize_final_comparison_plots__significance_comparison(tables['final_comparison_signals'], result['selections']),
             'final_significance_comparison')]


def finalize_final_comparison_plots_final_comparison_tables(result):
    """Return exact plotted values, with a stable 1–60 model-index mapping."""
    frame = result['tables']['vd_closure1d']
    paths = ((frame.study.eq('loose') & frame.scan_axis.eq('a_iso'))
             | (frame.study.eq('gap') & frame.scan_axis.eq('b_iso')))
    vr = frame.loc[paths & frame.mass_slice.eq('intermediate')
                   & frame.source.eq('DATA') & frame.plane.isin(['PP', 'PD', 'DP'])].copy()
    vr = vr.sort_values(['study', 'plane', 'scan_value']).reset_index(drop=True)
    if len(vr) != 30 or vr.duplicated(['study', 'plane', 'scan_value']).any():
        raise AssertionError('Require two intermediate approaches, three categories and five isolation edges.')
    hidden = ~vr.display_allowed.astype(bool)
    if not vr.loc[hidden, ['A', 'A_error', 'pred', 'pred_error', 'kappa', 'kappa_error']].isna().all().all():
        raise AssertionError('Hidden integrated VR values must remain NaN in the overlay.')
    for study, axis in [('loose', 'a_iso'), ('gap', 'b_iso')]:
        spec = result['selections'][study]
        for plane in ['PP', 'PD', 'DP']:
            part = vr.loc[vr.study.eq(study) & vr.plane.eq(plane)]
            if not finalize_final_comparison_plots_np.allclose(part.scan_value, [.2, .25, .3, .4, .5]):
                raise AssertionError('The overlay requires the same exact native isolation-edge grid.')
            chosen = part.loc[part.selected_point.astype(bool)]
            if len(chosen) != 1 or not finalize_final_comparison_plots_np.isclose(chosen.iloc[0].scan_value, spec[axis]):
                raise AssertionError('Each approach/category must retain its actual selected isolation edge.')
            for field in ['a_iso', 'a_phi', 'b_iso', 'c_phi']:
                if not finalize_final_comparison_plots_np.isclose(chosen.iloc[0][field], spec[field]):
                    raise AssertionError('Selected overlay point differs from the final geometry.')
            fixed = ['a_phi', 'c_phi'] if study == 'loose' else ['a_iso', 'a_phi', 'c_phi']
            if not all(finalize_final_comparison_plots_np.allclose(part[field], spec[field]) for field in fixed):
                raise AssertionError('Only the stated isolation edge may change along the comparison path.')
    scan = result['tables']['if_scan']
    signals = result['tables']['if_signals']
    selected = {}
    for study in ['loose', 'gap']:
        spec = result['selections'][study]
        point = scan.loc[finalize_final_comparison_plots_np.isclose(scan.a_iso, spec['a_iso']) & finalize_final_comparison_plots_np.isclose(scan.a_phi, spec['a_phi'])]
        if len(point) != 1:
            raise AssertionError('Each final A needs one exact nominal MC source point.')
        point = point.iloc[0]
        part = signals.loc[signals.candidate_id.eq(point.candidate_id) & signals.matching_decay.astype(bool)].copy()
        if len(part) != 60 or not part['sample'].is_unique:
            raise AssertionError('Require 60 distinct matching 2Mu2E hypotheses per final A.')
        if not finalize_final_comparison_plots_np.isfinite(part[list(finalize_final_comparison_plots_METRICS)].to_numpy(float)).all() or (part[list(finalize_final_comparison_plots_METRICS)] < 0).any().any():
            raise AssertionError('Absolute significance values must be finite and nonnegative.')
        for metric in finalize_final_comparison_plots_METRICS:
            finalize_final_comparison_plots_np.testing.assert_allclose(part[metric].median(), point[metric+'_median'], rtol=1e-10, atol=1e-12)
        part['approach'] = study
        part['final_candidate_id'] = spec['candidate_id']
        part['source_A_candidate_id'] = point.candidate_id
        selected[study] = part
    if set(selected['loose']['sample']) != set(selected['gap']['sample']):
        raise AssertionError('Both approaches must compare exactly the same signal hypotheses.')
    order = []
    for sample in selected['loose']['sample']:
        match = finalize_final_comparison_plots_re.fullmatch(r'2Mu2E_([\dp]+)GeV_([\dp]+)GeV_([\dp]+)mm', sample)
        if match is None:
            raise AssertionError('Unrecognized matching signal model: '+sample)
        parent, daughter, lifetime = [float(value.replace('p', '.')) for value in match.groups()]
        order.append(dict(sample=sample, parent_mass_GeV=parent, daughter_mass_GeV=daughter,
                          ctau_mm=lifetime))
    order = finalize_final_comparison_plots_pd.DataFrame(order).sort_values(['parent_mass_GeV', 'daughter_mass_GeV', 'ctau_mm']).reset_index(drop=True)
    order.insert(0, 'model_index', finalize_final_comparison_plots_np.arange(1, len(order)+1))
    if not finalize_final_comparison_plots_np.allclose(order.parent_mass_GeV.unique(), [200, 500, 800, 1000]):
        raise AssertionError('The final comparison requires the four established parent-mass groups.')
    if not order.groupby(['parent_mass_GeV', 'daughter_mass_GeV']).size().eq(5).all():
        raise AssertionError('Every parent/daughter group must contain the five separate lifetime models.')
    columns = ['approach', 'final_candidate_id', 'source_A_candidate_id', 'sample',
               'S_A', 'var_S_A', 'b', 'var_b', *finalize_final_comparison_plots_METRICS]
    comparison = finalize_final_comparison_plots_pd.concat([selected[study][columns].merge(order, on='sample', validate='one_to_one')
                            for study in ['loose', 'gap']], ignore_index=True)
    comparison = comparison.sort_values(['approach', 'model_index']).reset_index(drop=True)
    return dict(final_comparison_vr=vr, final_comparison_signals=comparison)


def finalize_final_comparison_plots__intermediate_overlay(rows, selections):
    fig, axes = finalize_final_comparison_plots__canvas('Intermediate VR: loose boundary and buffer on one isolation scan',
                        rows=3, cols=3, height=10.4, bottom=.25, top=.88)
    fig.subplots_adjust(hspace=.25)
    for j, plane in enumerate(['PP', 'PD', 'DP']):
        selected_notes = []
        for study in ['loose', 'gap']:
            part = rows.loc[rows.study.eq(study) & rows.plane.eq(plane)].sort_values('scan_value')
            x = part.scan_value.to_numpy(float)
            color = finalize_final_comparison_plots_COLORS[study]
            point = part.loc[part.selected_point.astype(bool)].iloc[0]
            axes[0, j].errorbar(x, part.A, yerr=part.A_error, fmt='o-', color=color,
                                ms=4, capsize=2, lw=1.1)
            axes[0, j].errorbar(x, part.pred, yerr=part.pred_error, fmt='s--', color=color,
                                mfc='white', ms=4, capsize=2, lw=1.1)
            finalize_final_comparison_plots__closure_points(axes[1, j], x, part, color)
            finalize_final_comparison_plots__closure_points(axes[1, j], [point.scan_value], part.loc[part.selected_point.astype(bool)],
                            color, marker='D', connect=False)
            axes[2, j].plot(x, 100*part.max_S_B, 'o-', color=color, ms=4, lw=1.1)
            axes[2, j].scatter([point.scan_value], [100*point.max_S_B], marker='D',
                               color=color, edgecolors='white', s=42, zorder=5)
            if point.display_allowed:
                selected_notes.append((color, f'{finalize_final_comparison_plots_LABELS[study]}: κ={point.kappa:.3f}±{point.kappa_error:.3f}'))
            else:
                selected_notes.append((color, finalize_final_comparison_plots_LABELS[study]+': selected DATA withheld'))
            missing = ~part.display_allowed.to_numpy(bool)
            for ax in axes[:2, j]:
                ax.scatter(x[missing], finalize_final_comparison_plots_np.full(missing.sum(), -.025), marker='x', s=20,
                           color=color, transform=ax.get_xaxis_transform(), clip_on=False)
        axes[0, j].set(title=plane, ylabel='DATA events', ylim=(0, None))
        finalize_final_comparison_plots__closure_axis(axes[1, j])
        for index, (color, text) in enumerate(selected_notes):
            axes[1, j].text(.025, .96-.10*index, text, transform=axes[1, j].transAxes,
                            va='top', fontsize=8, color=color,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=.5))
        axes[2, j].axhline(10, color='#A33333', ls='--', lw=1.1)
        axes[2, j].set(ylabel='Max ABCD S/B_MC [%]', yscale='log')
        for ax in axes[:, j]:
            ax.axvline(selections['loose']['a_iso'], color='#444444', ls=':', lw=1.2)
            ax.set(xticks=[.2, .25, .3, .4, .5], xlim=(.182, .518))
            ax.tick_params(axis='x', labelsize=8)
        axes[2, j].set_xlabel('Isolation edge: loose I_A / buffer I_H', fontsize=9)
    handles = [finalize_final_comparison_plots_Line2D([], [], color=finalize_final_comparison_plots_COLORS[s], lw=2, label=finalize_final_comparison_plots_LABELS[s]) for s in ['loose', 'gap']]
    handles += [finalize_final_comparison_plots_Line2D([], [], color='#444444', marker='o', label='Observed DATA A'),
                finalize_final_comparison_plots_Line2D([], [], color='#444444', marker='s', mfc='white', ls='--', label='Prediction B×C/D'),
                finalize_final_comparison_plots_Line2D([], [], color='#444444', ls=':', label='Selected isolation edge = 0.25'),
                finalize_final_comparison_plots_Patch(facecolor='#85B879', alpha=.25, label='κ=0.9–1.1 visual guide')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .125),
               ncol=3, frameon=False, fontsize=9)
    finalize_final_comparison_plots__caption(fig, '150≤MJJ<300 GeV; explicit PP/PD/DP categories; partner isolation<0.5. Color identifies the approach, while line/marker style distinguishes observed A from B×C/D.\n'
             'The shared x coordinate is an isolation edge, not an identical A definition: loose varies I_A at |Δφ|≥2.6; buffer varies I_H with A fixed at I<0.20, |Δφ|≥2.7 and C/D |Δφ|<2.0.\n'
             'Both final choices use isolation edge 0.25. First two rows use approved VR DATA and propagated statistical errors; the bottom row uses the maximum S/B_MC over ABCD and 120 separate hypotheses.\n'
             'Neighboring cuts and the two approaches share events; no independence is implied. The κ band is a visual guide, not a confidence interval. Nominal DATA A remains blinded.')
    return fig


def finalize_final_comparison_plots__significance_comparison(rows, selections):
    fig, axes = finalize_final_comparison_plots__canvas('Final approaches: absolute nominal MC-A significance by signal model',
                        cols=3, height=7.7, bottom=.30, top=.83)
    titles = [r'$S/\sqrt{S+B}$', 'Known-background Asimov Z', 'Finite-MC diagnostic Z']
    values = rows[list(finalize_final_comparison_plots_METRICS)].to_numpy(float)
    if (values <= 0).any():
        raise AssertionError('This logarithmic absolute-significance view requires positive values.')
    limits = (float(values.min())*.55, float(values.max())*2.0)
    parents = rows.loc[rows.approach.eq('loose')].groupby('parent_mass_GeV', sort=True)
    for ax, metric, title in zip(axes[0], finalize_final_comparison_plots_METRICS, titles):
        for number, (mass, part) in enumerate(parents):
            left, right = part.model_index.min()-.5, part.model_index.max()+.5
            if number % 2 == 0:
                ax.axvspan(left, right, color='#E8EFF8', alpha=.5, zorder=0)
            ax.text((left+right)/2, .98, f'{mass:g} GeV', ha='center', va='top',
                    transform=ax.get_xaxis_transform(), fontsize=8, color='#444444')
            if number:
                ax.axvline(left, color='#8795A7', lw=.9)
            for edge in [left+5, left+10]:
                ax.axvline(edge, color='#B6C0CB', ls=':', lw=.7)
        for study, marker in [('loose', 'o'), ('gap', '^')]:
            part = rows.loc[rows.approach.eq(study)].sort_values('model_index')
            ax.plot(part.model_index, part[metric], marker=marker, ls='none', color=finalize_final_comparison_plots_COLORS[study],
                    ms=3.4, alpha=.85, label=finalize_final_comparison_plots_LABELS[study])
        ax.set(ylabel='Absolute '+('S / √(S+B)' if metric=='s_sqrt_s_plus_b' else 'Z'),
               yscale='log', ylim=limits, xlim=(.5, 60.5), xticks=[1, 10, 20, 30, 40, 50, 60],
               xlabel='Signal hypothesis index')
        ax.set_title(title, pad=25, fontsize=13)
        ax.tick_params(axis='x', labelsize=8)
        medians = [rows.loc[rows.approach.eq(s), metric].median() for s in ['loose', 'gap']]
        ax.text(.5, 1.015, f'Medians: loose {medians[0]:.5f}; buffer {medians[1]:.5f}',
                transform=ax.transAxes, fontsize=8.5, va='bottom', ha='center')
    loose, gap = selections['loose'], selections['gap']
    handles = [finalize_final_comparison_plots_Line2D([], [], color=finalize_final_comparison_plots_COLORS['loose'], marker='o', ls='none',
                     label=f'Loose A: I<{loose["a_iso"]:.2f}, |Δφ|≥{loose["a_phi"]:g}'),
               finalize_final_comparison_plots_Line2D([], [], color=finalize_final_comparison_plots_COLORS['gap'], marker='^', ls='none',
                     label=f'Buffer A: I<{gap["a_iso"]:.2f}, |Δφ|≥{gap["a_phi"]:g}')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .185),
               ncol=2, frameon=False, fontsize=10)
    finalize_final_comparison_plots__caption(fig, 'Each x position is the same separate 2Mu2E signal hypothesis for both approaches. Values are absolute significance diagnostics on a logarithmic y axis; no WP2 normalization is applied.\n'
             'Models 1–60 are ordered by parent mass (200, 500, 800, 1000 GeV), then daughter mass (0.25, 1.2, 5 GeV), then increasing cτ within each five-model group. Exact mapping is exported in final_comparison_signals.csv.\n'
             'Markers are not a continuous mass/lifetime curve, and different signal hypotheses are never summed. S and B are direct nominal MC-A yields; the buffer uses the unchanged WP2 A.\n'
             'Finite-MC Z is the established MC-statistics diagnostic. These metrics omit the complete ABCD control likelihood and closure/transfer uncertainty, and are not observed DATA significances.')
    return fig


# ========================================================================
# abcd_4mu_final_09style_scans: original study API
# ========================================================================
"""Notebook-09 summary layouts filled only with notebook-10 4μ products.

No MC projection, DATA query, gate change or working-point selection occurs
here.  Red stars identify supplied comparison points, never an eligible optimum.
"""
import numpy as finalize_4mu_final_09style_scans_np
import pandas as finalize_4mu_final_09style_scans_pd
import matplotlib.pyplot as finalize_4mu_final_09style_scans_plt
from matplotlib.lines import Line2D as finalize_4mu_final_09style_scans_Line2D
from matplotlib.patches import Rectangle as finalize_4mu_final_09style_scans_Rectangle

finalize_4mu_final_09style_scans_reference = _FINALIZE_NAMESPACES['abcd_iso_buffer_plots']
finalize_4mu_final_09style_scans_gap_reference = _FINALIZE_NAMESPACES['abcd_capped_gap_plots']
finalize_4mu_final_09style_scans_comparison_reference = _FINALIZE_NAMESPACES['abcd_final_comparison_plots']
finalize_4mu_final_09style_scans_reference_canvas = _FINALIZE_NAMESPACES['abcd_final_check_plots']._canvas; finalize_4mu_final_09style_scans__caption = _FINALIZE_NAMESPACES['abcd_final_check_plots']._caption; finalize_4mu_final_09style_scans__style_table = _FINALIZE_NAMESPACES['abcd_final_check_plots']._style_table


finalize_4mu_final_09style_scans_ANGLES = finalize_4mu_final_09style_scans_reference.ANGLES
finalize_4mu_final_09style_scans_COLORS = finalize_4mu_final_09style_scans_reference.COLORS
finalize_4mu_final_09style_scans_METRICS = ('s_sqrt_s_plus_b', 'z_known_b', 'z_mc_stat_diagnostic')
finalize_4mu_final_09style_scans_COMPARISON_LABEL = 'Loose MC-A maximum (gate fails)'
finalize_4mu_final_09style_scans_BUFFER_COMPARISON_LABEL = 'WP2 buffer comparison (not selected)'


def finalize_4mu_final_09style_scans_make_scan_figures(result):
    """Return the fourteen non-VR pages using the original notebook-09 keys."""
    rows, buffers, loose, gap = finalize_4mu_final_09style_scans__inputs(result)
    buffer_reference = rows.loc[finalize_4mu_final_09style_scans_np.isclose(rows.a_iso,gap.a_iso)
                                & finalize_4mu_final_09style_scans_np.isclose(rows.a_phi,gap.a_phi)]
    if len(buffer_reference)!=1:
        raise AssertionError('The buffer needs its exact unchanged-A nominal MC reference.')
    buffer_reference = buffer_reference.iloc[0]
    plane = finalize_4mu_final_09style_scans_display_mc_plane(result['backgrounds']['nominal_DD'])
    model_rows, model_order = finalize_4mu_final_09style_scans__comparison_signals(result, loose, gap)
    figures, captions, definitions = [], {}, []

    def add(name, figure, section, question, caption):
        figures.append((name, figure))
        captions[name] = caption
        definitions.append(dict(name=name, section=section, question=question))

    add('if_nominal_plane', finalize_4mu_final_09style_scans__nominal_plane(plane, loose, gap), 'overview',
        'Where are the Reference, extended loose A and fixed WP2 A on the nominal MC plane used in 09?',
        'The full and zoom views follow 09 and show the Reference, extended loose comparison point and fixed WP2 boundaries. MC is 4μ nominal DD; '
        'the last displayed isolation bin includes the I≥1 tail, and angle flows are folded into the two end bins. '
        'The comparison point is not an adopted optimum; nominal Data are not included.')
    add('if_approaches', finalize_4mu_final_09style_scans__approaches(loose, gap), 'overview',
        'How do common ABCD boundaries differ from a buffer with fixed A?',
        'The two-panel schematic from 09 uses the actual 4μ comparison points. The left panel shows ordinary boundaries extending A; '
        'the right panel shows a buffer preserving the original WP2 A. The current isolation gap is zero, '
        'so only an angle buffer is present. Diagram dimensions are not proportional to the variable intervals.')
    add('if_tradeoff', finalize_4mu_final_09style_scans__tradeoff(rows, loose), 'loose',
        'How do MC-A sensitivity and control contamination vary in the loose isolation and angle scans?',
        'The three panels follow 09, with angle colors, circle-marked lines and dashed finite-MC Z curves. Sensitivity is the absolute median over 60 '
        'separate matching 4Mu models; contamination is the B/C/D maximum over all 120 models. '
        'The red star is the loose diagnostic point maximizing median MC-A known-Z before the gate, not an adopted optimum.')
    add('if_selection', finalize_4mu_final_09style_scans__selection(rows, loose), 'loose',
        'Does any point pass the nominal contamination gate in the capped loose scan at each angle?',
        'The table follows the angle-by-angle format of 09. The first passing point remains No pass, and the maximum MC-A point at each angle '
        'is shown as a diagnostic that fails the gate. Nominal Data controls are not queried.')
    closure = finalize_4mu_final_09style_scans_reference._nominal_closure({'tables': {'if_scan': rows}})
    finalize_4mu_final_09style_scans__channel(closure)
    add('if_nominal_closure', closure, 'loose',
        'What are the nominal MC factorization and sideband prediction statistics in the loose scan?',
        'Nominal MC κ and the relative BC/D error use the same plotting function as 09. The κ axis spans 0–2, '
        'and the green 0.9–1.1 band is a visual guide. Central values outside the view are retained as numbers and arrows.')
    add('gap_tradeoff', finalize_4mu_final_09style_scans__gap_tradeoff(buffers, gap, buffer_reference), 'buffer',
        'How does a fixed-A buffer affect MC-A sensitivity and control contamination?',
        'The three panels follow 09 and compare native 4μ φ_L=2.4,2.2,2.0,1.5. A is identical throughout the buffer scan, '
        'so direct-MC sensitivity remains constant. The red star marks the minimum-contamination comparison point; it does not imply a gate pass or adoption.')
    add('buffer2d_nominal', finalize_4mu_final_09style_scans__nominal_distribution(plane, gap), 'buffer',
        'Which parts of the nominal MC plane are used by fixed A and the buffer controls?',
        'The native MC full and zoom panels follow 09, with a shared color scale, red boundaries with white halos, and '
        'a yellow /// buffer. Display-bin flow folding and conservation of the original integrated sumw and sumw2 are checked.')
    add('gap_nominal_closure', finalize_4mu_final_09style_scans__gap_closure(buffers, gap), 'buffer',
        'How does nominal MC closure vary along the two buffer-boundary directions?',
        'MC κ is scanned along isolation and angle as in 09. A is fixed to 4μ WP2. '
        'Red diamonds and vertical lines mark comparison points; failing candidates are not labeled as optima.')
    support = finalize_4mu_final_09style_scans_gap_reference._gap_support(buffers, gap)
    finalize_4mu_final_09style_scans__channel(support)
    finalize_4mu_final_09style_scans__replace_caption(support,
        f'All panels use phi_L={gap.c_phi:g} and the same fixed 4mu WP2 A. The red vertical line is I_H={gap.b_iso:.2f}.\n'
        f'N_eff is a weighted-statistics diagnostic, not a raw count. C is unchanged by moving I_H and has {gap.neff_C:.2f} effective MC events.\n'
        'Direct MC A and MC B×C/D are shown with their existing propagated errors. Sparse control support and factorization are separate diagnostics.\n'
        'No nominal B/C/D gate passes. This supplied comparison is not selected, and no nominal DATA A or control count is used.')
    add('gap_support', support, 'buffer',
        'Are the weighted yields, effective statistics and prediction precision of the buffer controls adequate?',
        'The same 2×2 function as 09 shows ABCD weighted yields, N_eff, B/C/D contamination, MC A and '
        'MC BC/D. N_eff is not a raw-entry count. This is not a nominal Data prediction.')
    add('gap_selection', finalize_4mu_final_09style_scans__gap_selection(buffers, gap), 'buffer',
        'What limits the buffer candidates when statistical support and gate failures are considered together?',
        'The buffer-candidate table and highlighted row follow 09. The highlight is the supplied WP2 comparison point; '
        'all candidates explicitly retain their nominal gate failure and unevaluated control status. No new candidate is selected.')
    add('if_nominal_data_contamination', finalize_4mu_final_09style_scans__nominal_contamination(result, loose, gap), 'comparison',
        'What is the nominal B/C/D contamination for extended loose A and the fixed WP2 buffer?',
        'The MC-contamination panel and control-status table follow 09. Ratios use the existing all-120 envelope. '
        'Nominal Data B/C/D are unpublished, so — and withheld replace numerical values; A is not used. '
        'The current buffer B overlaps extended loose A, preventing a nominal Data comparison between the definitions.')
    significance = finalize_4mu_final_09style_scans_comparison_reference._significance_comparison(model_rows, {'loose': loose, 'gap': gap})
    finalize_4mu_final_09style_scans__channel(significance)
    finalize_4mu_final_09style_scans__replace_caption(significance,
        'Each index is the same separate matching 4Mu hypothesis in both approaches; absolute MC-A metrics are shown without reference-WP normalization.\n'
        'Models are ordered by m_B, then m_ZD and proper lifetime within each five-model group. Exact coordinates are retained in the model-order table.\n'
        f'The loose MC-A maximum uses A=(I<{loose.a_iso:g}, |Δφ|≥{loose.a_phi:g}); the buffer retains WP2 A=(I<{gap.a_iso:g}, |Δφ|≥{gap.a_phi:g}). Their A definitions differ.\n'
        'Contamination uses all 120 hypotheses. No candidate passes the nominal B/C/D gate; neither comparison is selected. No nominal DATA are used.')
    add('final_significance_comparison', significance, 'comparison',
        'How does direct-MC A sensitivity compare for the same 60 4Mu models in the same order?',
        'The three-panel plotting function is reused from 09. Blue circles show the extended loose A MC diagnostic; orange triangles show '
        'the buffer with fixed WP2 A. Each model is compared using the values for its two actual A regions. This is not full-likelihood sensitivity '
        'including control statistics, leakage and closure nuisances.')
    add('if_meeting_comparison', finalize_4mu_final_09style_scans__meeting_comparison(result, loose, gap), 'comparison',
        'What definitions, statistics and releasable VR results support assessment of the two comparison points?',
        'The evidence-comparison table follows 09. It lists the actual 4μ geometry, MC-A sensitivity, all-120 contamination and '
        'releasable VR κ only. Nominal controls and predictions remain unevaluated, and neither point is adopted. '
        'Overlap between buffer controls and extended loose A also limits a nominal Data comparison.')
    add('final_data_expected_significance', finalize_4mu_final_09style_scans__unavailable_data_significance(), 'comparison',
        'Can expected sensitivity using nominal Data sidebands be evaluated at present?',
        'The three-panel layout from 09 is retained, with the reason for not evaluating sensitivity: no nominal B/C/D gate-passing candidate '
        'or public counts are available. No arbitrary Data-background, prediction or significance curves are generated.')
    for _, figure in figures:
        finalize_4mu_final_09style_scans_plt.close(figure)
    conservation = []
    raw = result['backgrounds']['nominal_DD']
    for original, displayed, moment in (('values','MC','sumw'),('variances','variance_MC','sumw2')):
        before, after = float(raw[original][1:,:].sum()), float(plane[displayed].sum())
        finalize_4mu_final_09style_scans_np.testing.assert_allclose(before, after, rtol=1e-12, atol=1e-10)
        conservation.append(dict(scope='nominal_DD',moment=moment,before=before,after=after,passed=True))
    return dict(figures=figures,captions=captions,definitions=finalize_4mu_final_09style_scans_pd.DataFrame(definitions),
        tables=dict(style09_model_order=model_order,style09_comparison_signals=model_rows,
                    style09_comparison_candidates=result['plot_candidates'].copy(),
                    style09_display_conservation=finalize_4mu_final_09style_scans_pd.DataFrame(conservation)),
        metadata=dict(reference='Notebook 09 exact plotting helpers and page layouts',
            figure_count=len(figures),original_DATA_files_opened=0,numerical_logic_changed=False,
            working_point_selected=False,red_star_role=dict(loose=finalize_4mu_final_09style_scans_COMPARISON_LABEL,buffer=finalize_4mu_final_09style_scans_BUFFER_COMPARISON_LABEL),
            nominal_DATA_A_evaluated=False,nominal_DATA_BCD_evaluated=False))


def finalize_4mu_final_09style_scans_display_mc_plane(flowplane):
    """Fold the existing MC moments into 09's [1,2] tail-display convention."""
    xedges=finalize_4mu_final_09style_scans_np.asarray(flowplane['xedges'],float)
    yedges=finalize_4mu_final_09style_scans_np.asarray(flowplane['yedges'],float)
    matches=finalize_4mu_final_09style_scans_np.flatnonzero(finalize_4mu_final_09style_scans_np.isclose(xedges,1.))
    if len(matches)!=1:
        raise AssertionError('The 09 display needs the existing exact isolation edge at 1.')
    tail=int(matches[0])
    display=dict(xedges=finalize_4mu_final_09style_scans_np.r_[xedges[:tail+1],2.],yedges=yedges.copy())
    for original,field in (('values','MC'),('variances','variance_MC')):
        source=finalize_4mu_final_09style_scans_np.asarray(flowplane[original],float)
        if source.shape!=(len(xedges)+1,len(yedges)+1):
            raise AssertionError('Expected the existing explicit-flow MC plane.')
        supported=source[1:,:]
        folded=supported[:,1:-1].copy()
        folded[:,0]+=supported[:,0]
        folded[:,-1]+=supported[:,-1]
        displayed=finalize_4mu_final_09style_scans_np.concatenate((folded[:tail],folded[tail:].sum(axis=0,keepdims=True)),axis=0)
        finalize_4mu_final_09style_scans_np.testing.assert_allclose(displayed.sum(),supported.sum(),rtol=1e-12,atol=1e-10)
        display[field]=displayed
    return display


def finalize_4mu_final_09style_scans__inputs(result):
    tables=result['tables']
    scan=tables['mc_scan'].loc[tables['mc_scan'].scope.eq('nominal_DD')].copy()
    if scan.purity_pass_BCD.any():
        raise AssertionError('These unselected comparison pages require the recorded zero-pass state.')
    if not scan.signal_hypotheses.eq(120).all():
        raise AssertionError('Retain the all-120 contamination envelope.')
    rows=scan.loc[scan.strategy.eq('loose')].copy()
    if len(rows)!=20 or not rows.groupby('a_phi').size().eq(5).all():
        raise AssertionError('Retain the twenty native ordinary-boundary candidates.')
    rows['purity_pass']=rows.purity_pass_BCD
    specifications={}
    for group in ('loose_extended','buffer_wp2'):
        part=result['plot_candidates'].loc[result['plot_candidates'].group.eq(group)]
        if len(part)!=1:
            raise AssertionError('Require one supplied extended-loose point and one fixed-WP2 buffer point.')
        identity=part.iloc[0].candidate_id
        specifications[group]=scan.loc[scan.candidate_id.eq(identity)].iloc[0].copy()
    loose,gap=specifications['loose_extended'],specifications['buffer_wp2']
    finalize_4mu_final_09style_scans_np.testing.assert_allclose(loose.z_known_b_median,rows.z_known_b_median.max(),rtol=1e-12,atol=1e-12)
    buffers=scan.loc[scan.strategy.eq('buffer_wp2') & scan.main_scan].copy()
    buffers['purity_pass']=buffers.purity_pass_BCD
    regions=tables['mc_regions'].loc[tables['mc_regions'].scope.eq('nominal_DD')]
    maxima=regions.pivot(index='candidate_id',columns='region',values='max_S_B')
    for region in 'BCD':
        buffers['max_S_B_'+region]=buffers.candidate_id.map(maxima[region])
    nominal=tables.get('data_closure',finalize_4mu_final_09style_scans_pd.DataFrame())
    if len(nominal):
        nominal=nominal.loc[nominal.scope.eq('nominal_DD')]
        fields=['A','B','C','D','pred','kappa']
        if nominal.display_allowed.any() or not nominal[fields].isna().all().all():
            raise AssertionError('Nominal DATA must remain unavailable in this zero-pass study.')
    return rows,buffers,loose,gap


def finalize_4mu_final_09style_scans__canvas(title,**kwargs):
    figure,axes=finalize_4mu_final_09style_scans_reference_canvas(title,**kwargs)
    finalize_4mu_final_09style_scans__channel(figure)
    return figure,axes


def finalize_4mu_final_09style_scans__channel(figure):
    title=figure._suptitle
    if title is not None:
        title.set_text(title.get_text().replace('2μ2e · ','4μ · ',1))


def finalize_4mu_final_09style_scans__replace_caption(figure,text):
    # Notebook 09 reserves this exact footer location for its wrapped caption.
    for artist in list(figure.texts):
        if finalize_4mu_final_09style_scans_np.allclose(artist.get_position(),(.5,.018)):
            artist.remove()
    finalize_4mu_final_09style_scans__caption(figure,text)


def finalize_4mu_final_09style_scans__nominal_plane(plane,loose,gap):
    fig,axes=finalize_4mu_final_09style_scans__canvas('Nominal background MC: native bins and visible boundary choices',cols=2,height=9.,bottom=.28,top=.83)
    fig.subplots_adjust(right=.93,wspace=.35)
    norm,_=finalize_4mu_final_09style_scans_reference._norm([plane['MC']])
    definitions=[(.3,2.6,'#0072B2','--','Reference (0.30, 2.6)'),
                 (gap.a_iso,gap.a_phi,'#D55E00','-',f'Fixed WP2 A ({gap.a_iso:.2f}, {gap.a_phi:g})'),
                 (loose.a_iso,loose.a_phi,'#C02672',':',f'Extended loose comparison ({loose.a_iso:g}, {loose.a_phi:g})')]
    for ax,xmax,title in zip(axes[0],(2.,.6),('Full display: isolation 0–2','Near-boundary view: isolation 0–0.6')):
        mesh=finalize_4mu_final_09style_scans_reference._plane_image(ax,plane,'MC',norm,None,xmax,title)
        for iso,phi,color,style,_ in definitions:
            finalize_4mu_final_09style_scans_reference._cut(ax,iso,True,color,style,2.6)
            finalize_4mu_final_09style_scans_reference._cut(ax,phi,False,color,style,2.6)
        ax.set_yticks([0,1,2,2.4,2.6,2.7,finalize_4mu_final_09style_scans_np.pi],labels=['0','1','2','2.4','2.6','2.7','π'])
        fig.colorbar(mesh,ax=ax,fraction=.046,pad=.025,label='Background MC events / display bin')
    fig.legend(handles=[finalize_4mu_final_09style_scans_Line2D([],[],color=color,ls=style,lw=2.6,label=label)
        for _,_,color,style,label in definitions],loc='lower center',bbox_to_anchor=(.5,.145),ncol=3,frameon=False,fontsize=9)
    finalize_4mu_final_09style_scans__caption(fig,'Nominal DD, MJJ≥150 GeV, subleading muon-LJ isolation in [0,0.5). Native MC counts use the identical notebook-09 display convention.\n'
        f'Reference is (0.30,2.6); fixed WP2 A is ({gap.a_iso:g},{gap.a_phi:g}); the loose MC-A maximum is ({loose.a_iso:g},{loose.a_phi:g}). No candidate is selected.\n'
        'Both panels share a logarithmic colour scale. The final [1,2] display bin contains all I≥1 tails/overflow; angular flows are folded. Sumw and sumw2 are conserved.\n'
        'The right panel changes only the view. No nominal DATA plane, A count, control count or closure is evaluated.')
    return fig


def finalize_4mu_final_09style_scans__approaches(loose,gap):
    fig,axes=finalize_4mu_final_09style_scans__canvas('Two approaches: common boundaries or fixed A with separated sidebands',cols=2,height=9.,bottom=.27,top=.84)
    for ax in axes[0]:
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Muon-LJ isolation',ylabel='|Δφ|',xticks=[],yticks=[])
        ax.grid(False)
    left,right=axes[0]
    x,y=.55,.52
    for name,xx,yy,width,height,color in [('A',0,y,x,1-y,'#FBE5E5'),('B',x,y,1-x,1-y,'#E8EFF8'),
            ('C',0,0,x,y,'#E8EFF8'),('D',x,0,1-x,y,'#E8EFF8')]:
        left.add_patch(finalize_4mu_final_09style_scans_Rectangle((xx,yy),width,height,facecolor=color,edgecolor='#8795A7'))
        tx,ty=(.42,.64) if name=='A' else (xx+width/2,yy+height/2)
        left.text(tx,ty,name,ha='center',va='center',fontsize=25,weight='bold')
    iso_moves=not finalize_4mu_final_09style_scans_np.isclose(loose.a_iso,gap.a_iso)
    phi_moves=not finalize_4mu_final_09style_scans_np.isclose(loose.a_phi,gap.a_phi)
    original_x=.30 if iso_moves else x
    original_y=.70 if phi_moves else y
    left.add_patch(finalize_4mu_final_09style_scans_Rectangle((0,original_y),original_x,1-original_y,
                            fill=False,edgecolor='#D55E00',ls='--',lw=2))
    left.text(original_x/2,(original_y+1)/2,'Original WP2 A',ha='center',fontsize=9,color='#9F4200')
    if iso_moves:
        left.annotate('',xy=(x-.02,.94),xytext=(original_x+.01,.94),
                      arrowprops=dict(arrowstyle='->',lw=2,color='#9F2929'))
    if phi_moves:
        left.annotate('',xy=(.15,y+.02),xytext=(.15,original_y-.01),
                      arrowprops=dict(arrowstyle='->',lw=2,color='#9F2929'))
    left.set_xticks([original_x,x] if iso_moves else [x],
                   [f'{gap.a_iso:.2f} (WP2)',f'{loose.a_iso:.2f} (loose)'] if iso_moves else [f'{loose.a_iso:.2f} (both)'],fontsize=9)
    left.set_yticks([y,original_y] if phi_moves else [y],
                   [f'{loose.a_phi:g} (loose)',f'{gap.a_phi:g} (WP2)'] if phi_moves else [f'{loose.a_phi:g} (same angle)'],fontsize=9)
    left.set_title('Loosen isolation: A expands right' if not phi_moves else
                   'Loosen both boundaries: A expands right and down',fontsize=13)
    left.text(.5,-.13,f'Loose MC-A comparison: I_A {gap.a_iso:.2f} → {loose.a_iso:.2f}; |Δφ| {gap.a_phi:g} → {loose.a_phi:g}.\n'
              'B/C/D share the updated ordinary boundaries.',transform=left.transAxes,ha='center',va='top',fontsize=10)
    right.set_facecolor('#eeeeee')
    # This supplied point has no isolation gap; do not draw a fictitious strip.
    xlo=.40
    xhi=xlo if finalize_4mu_final_09style_scans_np.isclose(gap.a_iso,gap.b_iso) else .57
    ylo,yhi=.40,.70
    for name,xx,yy,width,height,color in [('A',0,yhi,xlo,1-yhi,'#FBE5E5'),('B',xhi,yhi,1-xhi,1-yhi,'#E8EFF8'),
            ('C',0,0,xlo,ylo,'#E8EFF8'),('D',xhi,0,1-xhi,ylo,'#E8EFF8')]:
        right.add_patch(finalize_4mu_final_09style_scans_Rectangle((xx,yy),width,height,facecolor=color,edgecolor='#8795A7'))
        right.text(xx+width/2,yy+height/2,name,ha='center',va='center',fontsize=25,weight='bold')
    right.text(xlo/2,.76,'WP2 unchanged',fontsize=8.5,ha='center',color='#9F2929')
    right.text(.73,.55,'buffer',ha='center',va='center',fontsize=13,color='#666666')
    if xhi>xlo:right.text((xlo+xhi)/2,.55,'buffer',rotation=90,ha='center',va='center',fontsize=13,color='#666666')
    right.set_title('Buffer: WP2 A fixed; C/D move away in angle',fontsize=13)
    right.text(.5,-.09,f'A: I<{gap.a_iso:.2f}, φ≥{gap.a_phi:g}; B: I≥{gap.b_iso:.2f}, φ≥{gap.a_phi:g}\n'
               f'C: I<{gap.a_iso:.2f}, φ<{gap.c_phi:g}; D: I≥{gap.b_iso:.2f}, φ<{gap.c_phi:g}',
               transform=right.transAxes,ha='center',va='top',fontsize=10)
    finalize_4mu_final_09style_scans__caption(fig,'Conceptual diagram: display sizes are not proportional to physical cuts. Left is the supplied MC-A maximum before the failed control gate, not an eligible choice.\n'
        f'The ordinary loose A expands to I<{loose.a_iso:g}, |Δφ|≥{loose.a_phi:g}; controls share its updated edges. The right-hand buffer keeps the original WP2 A.\n'
        f'The buffer preserves A at I<{gap.a_iso:g}, |Δφ|≥{gap.a_phi:g}. Its isolation buffer is zero here; the excluded angular strip is {gap.c_phi:g}≤|Δφ|<{gap.a_phi:g}.\n'
        'The nominal B/C/D contamination gate fails for every candidate. MC-A diagnostics do not include the complete control-count or transfer likelihood.\n'
        'No nominal DATA are evaluated; this diagram introduces no new event selection or disclosure.')
    return fig


def finalize_4mu_final_09style_scans__tradeoff(rows,chosen):
    fig,axes=finalize_4mu_final_09style_scans__canvas('Relax isolation: absolute sensitivity versus sideband contamination',cols=3,height=7.8,bottom=.30,top=.82)
    for angle in finalize_4mu_final_09style_scans_ANGLES:
        part=rows.loc[finalize_4mu_final_09style_scans_np.isclose(rows.a_phi,angle)].sort_values('a_iso');color=finalize_4mu_final_09style_scans_COLORS[angle]
        axes[0,0].plot(part.a_iso,part.s_sqrt_s_plus_b_median,'-o',color=color,ms=5,label=f'|Δφ| boundary {angle:g}')
        axes[0,1].plot(part.a_iso,part.z_known_b_median,'-o',color=color,ms=5)
        axes[0,1].plot(part.a_iso,part.z_mc_stat_diagnostic_median,'--',color=color,alpha=.65)
        axes[0,2].plot(part.a_iso,100*part.max_S_B_BCD,'-o',color=color,ms=5)
        passed=part.purity_pass.astype(bool)
        axes[0,2].scatter(part.loc[passed,'a_iso'],100*part.loc[passed,'max_S_B_BCD'],
                          s=110,facecolors='none',edgecolors='#2D8B45',lw=1.6,zorder=4)
    for ax,value in zip(axes[0],[chosen.s_sqrt_s_plus_b_median,chosen.z_known_b_median,100*chosen.max_S_B_BCD]):
        ax.scatter([chosen.a_iso],[value],marker='*',s=190,color='#B32424',edgecolors='white',zorder=6)
    axes[0,0].set(title='S / √(S+B)',ylabel='Absolute median S / √(S+B)')
    axes[0,1].set(title='Asimov significance Z',ylabel='Absolute median Z')
    axes[0,2].set(title='Worst nominal B/C/D contamination',ylabel='Maximum S/B_MC over 120 hypotheses [%]',yscale='log')
    axes[0,2].axhline(10,color='#A33333',ls='--',lw=1.3)
    for ax in axes[0]:finalize_4mu_final_09style_scans_reference._iso_axis(ax,rows.a_iso)
    handles=[finalize_4mu_final_09style_scans_Line2D([],[],color=finalize_4mu_final_09style_scans_COLORS[a],label=f'|Δφ|={a:g}') for a in finalize_4mu_final_09style_scans_ANGLES]
    handles += [finalize_4mu_final_09style_scans_Line2D([],[],color='#444444',ls='--',label='Dashed Z: finite-MC diagnostic'),
        finalize_4mu_final_09style_scans_Line2D([],[],marker='*',color='#B32424',ls='none',ms=10,label=finalize_4mu_final_09style_scans_COMPARISON_LABEL),
        finalize_4mu_final_09style_scans_Line2D([],[],marker='o',mfc='none',mec='#2D8B45',ls='none',ms=8,label='BCD MC contamination <10%')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.135),ncol=4,frameon=False,fontsize=8.5)
    finalize_4mu_final_09style_scans__caption(fig,'The grid is I=0.20,0.25,0.30,0.40,0.50 and native |Δφ|=2.4,2.6,2.7,2.8; these are the unchanged notebook-10 MC candidates.\n'
        'Absolute per-hypothesis sensitivity metrics are summarized over the same 60 matching 4Mu models. Contamination uses 120 separate generated hypotheses.\n'
        'No candidate passes the central B/C/D S/B<10% gate. The red star maximizes nominal MC-A known-Z before that gate, and is not an eligible optimum.\n'
        'MC-A diagnostics omit the full ABCD control likelihood, leakage and transfer uncertainty. Neighbouring cuts share events; no nominal DATA are used.')
    return fig


def finalize_4mu_final_09style_scans__selection(rows,chosen):
    fig,axes=finalize_4mu_final_09style_scans__canvas('Loose candidate selection: no nominal B/C/D contamination pass',cols=1,height=7.8,bottom=.25,top=.85)
    ax=axes[0,0];ax.axis('off')
    contents=[]
    for angle in finalize_4mu_final_09style_scans_ANGLES:
        part=rows.loc[finalize_4mu_final_09style_scans_np.isclose(rows.a_phi,angle)].sort_values(
            ['z_known_b_median','a_iso'],ascending=[False,True])
        best=part.iloc[0]
        contents.append([f'{angle:g}','No pass ≤0.5',f'{best.a_iso:g}',
            f'{100*best.max_S_B_BCD:.3f}%',f'{best.s_sqrt_s_plus_b_median:.4f}',
            f'{best.z_known_b_median:.4f}',f'{best.z_mc_stat_diagnostic_median:.4f}',
            'MC comparison only' if best.candidate_id==chosen.candidate_id else 'MC maximum; fails gate'])
    table=ax.table(cellText=contents,colLabels=['Fixed |Δφ|','First passing iso','Best MC-A iso\n(before gate)',
        'Maximum BCD S/B','Median S/√(S+B)','Median known-B Z','Median finite-MC Z','Final choice'],cellLoc='center',
        colWidths=[.08,.12,.10,.14,.15,.15,.12,.14],bbox=[0,.30,1,.64]);finalize_4mu_final_09style_scans__style_table(table,9.3)
    fig.text(.075,.33,'No nominal B/C/D contamination-passing point exists on the capped 4mu grid. No working point is selected.\n'
        f'Extended loose MC-A maximum: ({chosen.a_iso:g},{chosen.a_phi:g}), median known-Z={chosen.z_known_b_median:.4f}; '
        f'maximum BCD S/B={100*chosen.max_S_B_BCD:.3f}%.',fontsize=11,weight='bold',color='#25354A')
    fig.text(.075,.21,'MC maxima in this table and the loose star are evaluated before the failed control gate; they are diagnostic comparisons only.\n'
        'Nominal B/C/D DATA, raw background prediction and DATA-based expected significance remain unavailable.',fontsize=10.5,color='#8D3429')
    finalize_4mu_final_09style_scans__caption(fig,'The angle-by-angle table has the original notebook-09 layout. MC-A maxima are listed before purity; the first-passing-point column remains empty of eligible candidates.\n'
        'Sensitivity remains an absolute median over 60 matching 4Mu hypotheses; contamination uses all 120 separate hypotheses.\n'
        'Neither VR agreement nor an attractive MC-A Z overrides the nominal contamination gate. Nominal DATA A remains blinded.')
    return fig


def finalize_4mu_final_09style_scans__gap_tradeoff(rows,chosen,reference_point):
    for metric in finalize_4mu_final_09style_scans_METRICS:
        finalize_4mu_final_09style_scans_np.testing.assert_allclose(rows[metric+'_median'],reference_point[metric+'_median'],rtol=1e-12,atol=1e-12)
    fig,axes=finalize_4mu_final_09style_scans__canvas('Buffer scan: fixed WP2 A, isolation sideband edge at most 0.5',cols=3,height=7.6,top=.83,bottom=.28)
    for color,phi in zip(['#0072B2','#D55E00','#009E73','#CC79A7'],[2.4,2.2,2.,1.5]):
        part=rows.loc[finalize_4mu_final_09style_scans_np.isclose(rows.c_phi,phi)].sort_values('b_iso')
        axes[0,0].plot(part.b_iso,finalize_4mu_final_09style_scans_np.full(len(part),reference_point.s_sqrt_s_plus_b_median),'o-',color=color,ms=4,label=f'C/D angle upper edge = {phi:g}')
        axes[0,1].plot(part.b_iso,part.z_known_b_median,'o-',color=color,ms=4)
        axes[0,1].plot(part.b_iso,part.z_mc_stat_diagnostic_median,'--',color=color,alpha=.65)
        axes[0,2].plot(part.b_iso,100*part.max_S_B_BCD,'o-',color=color,ms=5)
        good=part.purity_pass
        axes[0,2].scatter(part.loc[good,'b_iso'],100*part.loc[good,'max_S_B_BCD'],s=110,facecolors='none',edgecolors='#248342',lw=1.6)
    axes[0,0].set(title=r'$S/\sqrt{S+B}$ in MC A',ylabel='Median S / √(S+B) (absolute)',ylim=(0,1.2*reference_point.s_sqrt_s_plus_b_median))
    axes[0,1].set(title='Asimov significance Z in MC A',ylabel='Median Z (absolute)',ylim=(0,1.2*reference_point.z_known_b_median))
    axes[0,2].set(title='Nominal control contamination',ylabel='Maximum B/C/D S/B_MC [%]',yscale='log')
    axes[0,2].axhline(10.,color='#B32424',ls='--')
    for index,ax in enumerate(axes[0]):
        ax.set(xticks=[.2,.25,.3,.4,.5],xlim=(.185,.515),xlabel='B/D isolation lower edge I_H')
        value=[reference_point.s_sqrt_s_plus_b_median,chosen.z_known_b_median,100*chosen.max_S_B_BCD][index]
        ax.scatter([chosen.b_iso],[value],marker='*',s=200,color='#B32424',edgecolors='white',zorder=5)
    handles,_=axes[0,0].get_legend_handles_labels()
    handles += [finalize_4mu_final_09style_scans_Line2D([],[],color='#444444',ls='--',label='Dashed Z: finite-MC diagnostic'),
        finalize_4mu_final_09style_scans_Line2D([],[],marker='o',mfc='none',mec='#248342',ls='none',label='MC B/C/D contamination <10%'),
        finalize_4mu_final_09style_scans_Line2D([],[],marker='*',color='#B32424',ls='none',label=finalize_4mu_final_09style_scans_BUFFER_COMPARISON_LABEL)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.13),ncol=3,frameon=False,fontsize=9)
    finalize_4mu_final_09style_scans__caption(fig,f'A is identical at every point: I<{chosen.a_iso:.2f}, |Δφ|≥{chosen.a_phi:g}. The direct MC-A signal and background do not change.\n'
        'Absolute MC-A medians use 60 matching 4Mu hypotheses. The native 4mu control-edge curves are phi_L=2.4,2.2,2.0,1.5; no unavailable edge is interpolated.\n'
        'Right: maximum over 120 hypotheses and B/C/D. Larger I_H excludes more isolation sideband; phi_L is a control edge, not the A boundary.\n'
        'No buffer passes the nominal contamination gate. The star illustrates the supplied WP2 comparison only; it is not a selected minimum-change solution.')
    return fig


def finalize_4mu_final_09style_scans__nominal_distribution(plane,geometry):
    fig,axes=finalize_4mu_final_09style_scans__canvas('Fixed-A buffer in nominal background MC',cols=2,height=8.9,bottom=.27,top=.83)
    fig.subplots_adjust(right=.93,wspace=.35);norm,_=finalize_4mu_final_09style_scans_reference._norm([plane['MC']])
    for ax,xmax,title in zip(axes[0],(2.,.6),('Full display: isolation 0–2','Near-boundary view: isolation 0–0.6')):
        mesh=finalize_4mu_final_09style_scans_reference._plane_image(ax,plane,'MC',norm,geometry,xmax,title)
        fig.colorbar(mesh,ax=ax,fraction=.046,pad=.025,label='Background MC events / display bin')
    finalize_4mu_final_09style_scans_reference._distribution_legend(fig,data=False,gap=True)
    finalize_4mu_final_09style_scans__caption(fig,f'Nominal 4mu MC only. A: I<{geometry.a_iso:.2f}, φ≥{geometry.a_phi:g}; B: I≥{geometry.b_iso:.2f}, φ≥{geometry.a_phi:g}; '
        f'C: I<{geometry.a_iso:.2f}, φ<{geometry.c_phi:g}; D: I≥{geometry.b_iso:.2f}, φ<{geometry.c_phi:g}.\n'
        'The full and zoom panels use identical native counts and logarithmic colour scales; yellow hatching marks the actual angular buffer. The isolation gap is zero here.\n'
        'The final [1,2] display bin includes I≥1 tails/overflow; angular flows are folded; isolation underflow is excluded. Sumw and sumw2 are preserved.\n'
        'White-halo red boundaries and all panel settings reuse notebook 09. No nominal DATA or new working-point selection enters this figure.')
    return fig


def finalize_4mu_final_09style_scans__gap_closure(rows,geometry):
    chosen=rows.loc[rows.candidate_id.eq(geometry.candidate_id)]
    fig,axes=finalize_4mu_final_09style_scans__canvas('Buffer sideband starts: nominal MC closure with A fixed',cols=2,height=8.6,bottom=.26,top=.84)
    specifications=[('b_iso','c_phi',[1.5,2.,2.4]),('c_phi','b_iso',[.25,.3,.5])]
    for ax,(axis,fixed,values) in zip(axes[0],specifications):
        for level,(color,value) in enumerate(zip(['#0072B2','#D55E00','#009E73'],values)):
            part=rows.loc[finalize_4mu_final_09style_scans_np.isclose(rows[fixed],value)].sort_values(axis)
            if len(part):finalize_4mu_final_09style_scans_reference._closure_points(ax,part[axis],part,color,level=level,label=f'{"φ_L" if fixed=="c_phi" else "I_H"}={value:g}')
        finalize_4mu_final_09style_scans_reference._closure_points(ax,chosen[axis],chosen,'#B32424',marker='D',connect=False,label=finalize_4mu_final_09style_scans_BUFFER_COMPARISON_LABEL)
        ax.axvline(geometry[axis],color='#B32424',ls=':',lw=1)
        finalize_4mu_final_09style_scans_reference._closure_axis(ax);ax.legend(frameon=False,fontsize=9)
        ax.set_xticks(sorted(rows[axis].unique()));ax.tick_params(axis='x',labelsize=8)
        ax.set_xlabel('B/D isolation start I_H (≤0.5)' if axis=='b_iso' else 'C/D angle upper edge φ_L [rad]')
        ax.set_title('Move B/D away in isolation' if axis=='b_iso' else 'Move C/D away in angle')
    finalize_4mu_final_09style_scans__caption(fig,f'A stays fixed at I<{geometry.a_iso:.2f}, |Δφ|≥{geometry.a_phi:g}. Red marks the supplied comparison I_H={geometry.b_iso:.2f}, phi_L={geometry.c_phi:g}, not a passing solution.\n'
        'The panels scan control starts rather than the A boundary. Every plotted isolation control edge is at most 0.5; all shown edges are native 4mu boundaries.\n'
        'Nominal MC only. Kappa uses 0–2 with labelled off-frame central values and a 0.9–1.1 visual guide, not a confidence interval.\n'
        'First-order weighted-MC ratio errors and correlated neighbouring cuts are diagnostics. No nominal DATA A or DATA closure is evaluated.')
    return fig


def finalize_4mu_final_09style_scans__gap_selection(rows,chosen):
    near=rows.loc[finalize_4mu_final_09style_scans_np.isclose(rows.c_phi,chosen.c_phi)&rows.b_iso.le(.4)].sort_values('b_iso')
    iso_edges=[.25,.3,.4,.5,.75,1.,2.];phi_edges=[.5,1.,1.5,2.,2.2,2.4]
    contents=[]
    for row in near.itertuples():
        movement=iso_edges.index(float(row.b_iso))-iso_edges.index(float(row.a_iso))+phi_edges.index(float(row.a_phi))-phi_edges.index(float(row.c_phi))
        contents.append([f'{row.b_iso:g}',f'{row.c_phi:g}',str(movement),f'{100*row.max_S_B_BCD:.3f}%',
            f'{row.min_sideband_neff:.2f}',f'{row.pred:.2f} ± {row.pred_error:.2f}','YES' if row.purity_pass else 'NO'])
    fig,axes=finalize_4mu_final_09style_scans__canvas('Buffer comparison: control support and no eligible nominal solution',cols=1,height=7.8,top=.85,bottom=.30)
    ax=axes[0,0];ax.axis('off')
    table=ax.table(cellText=contents,colLabels=['I_H','phi_L','Native movement','Max BCD S/B_MC','Min BCD N_eff','MC B×C/D','BCD gate'],
        cellLoc='center',colWidths=[.08,.08,.15,.18,.15,.21,.15],bbox=[0,.30,1,.65]);finalize_4mu_final_09style_scans__style_table(table,10)
    index=int(finalize_4mu_final_09style_scans_np.flatnonzero(near.candidate_id.eq(chosen.candidate_id).to_numpy())[0])+1
    for column in range(7):table[(index,column)].set_facecolor('#FFF0DC')
    fig.text(.075,.37,f'WP2 comparison only: A is I<{chosen.a_iso:.2f}, |Δφ|≥{chosen.a_phi:g}; B/D I≥{chosen.b_iso:.2f}; C/D |Δφ|<{chosen.c_phi:g}.\n'
        'No buffer passes nominal B/C/D contamination. DATA controls and raw DATA B×C/D are not evaluated.',fontsize=12,weight='bold',color='#25354A')
    fig.text(.075,.24,'The highlighted row is the supplied minimum-contamination illustration within the main WP2 grid.\n'
        'It is not a selected optimum; VR checks and MC-A sensitivity cannot replace the failed nominal control gate.',fontsize=11,color='#8D3429')
    finalize_4mu_final_09style_scans__caption(fig,'The table and row highlight use the original notebook-09 layout. Native movement counts existing boundary steps and is not a physics-efficiency measure.\n'
        'All points preserve WP2 A and its MC-A sensitivity. C and other controls may have very sparse weighted-MC support.\n'
        'There is no eligible minimum-change solution to report. Nominal DATA A, B/C/D counts and a DATA background prediction remain unavailable.')
    return fig


def finalize_4mu_final_09style_scans__nominal_contamination(result,loose,gap):
    regions=result['tables']['mc_regions']
    regions=regions.loc[regions.scope.eq('nominal_DD')]
    fig,axes=finalize_4mu_final_09style_scans__canvas('Nominal B/C/D: MC contamination and control counts',cols=1,height=7.4,bottom=.46,top=.84)
    fig.set_size_inches(13.5,7.4);ax=axes[0,0]
    positive,contents,handles=[],[],[]
    for point,shift,color,label in zip((loose,gap),(-.12,.12),('#0072B2','#D55E00'),('Extended loose A','Buffer WP2')):
        part=regions.loc[regions.candidate_id.eq(point.candidate_id)].set_index('region').loc[list('BCD')]
        values=100*part.max_S_B.to_numpy(float)
        ax.plot(finalize_4mu_final_09style_scans_np.arange(3)+shift,values,'s',color=color,mfc='white',ms=6)
        positive.extend(values[finalize_4mu_final_09style_scans_np.isfinite(values)&(values>0)])
        status='DATA withheld: gate / A overlap' if point.candidate_id==gap.candidate_id else 'DATA withheld: gate fails'
        contents.append([label+' (not selected)','—','—','—',f'{values.max():.3f}%',status])
        handles.append(finalize_4mu_final_09style_scans_Line2D([],[],color=color,marker='s',mfc='white',ls='none',label=label+' comparison'))
    ax.set(title='Central MC contamination gate',ylabel='Maximum S/B_MC over 120 hypotheses [%]',yscale='log',
        xticks=finalize_4mu_final_09style_scans_np.arange(3),xticklabels=['B · high isolation','C · low |Δφ|','D · both sidebands'],
        ylim=(min(positive)*.55,max(20.,max(positive)*1.7)))
    ax.axhline(10,color='#A33333',ls='--',lw=1.2);ax.tick_params(axis='x',labelsize=9)
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.365),ncol=2,frameon=False,fontsize=9)
    table_ax=fig.add_axes([.075,.13,.895,.19]);table_ax.axis('off')
    table=table_ax.table(cellText=contents,colLabels=['Definition','DATA B','DATA C','DATA D','Max BCD S/B_MC','Nominal DATA status'],
        cellLoc='center',colWidths=[.29,.085,.085,.085,.18,.275],bbox=[0,0,1,1]);finalize_4mu_final_09style_scans__style_table(table,9.2)
    finalize_4mu_final_09style_scans__caption(fig,'The MC gate is the maximum S/B_MC over 120 separate hypotheses in each control region; the dashed line is the unchanged 10% central-value threshold.\n'
        'No nominal 4mu candidate passes. The table therefore contains no observed B/C/D counts and no inferred zero. Different signal models are never summed.\n'
        'Buffer controls overlap the extended loose A; nominal DATA comparison is unavailable. Both points are diagnostics, not selected optima. No new DATA query occurs.')
    return fig


def finalize_4mu_final_09style_scans__comparison_signals(result,loose,gap):
    signals=result['tables']['mc_signals']
    signals=signals.loc[signals.scope.eq('nominal_DD')&signals.matching_decay]
    first=signals.loc[signals.candidate_id.eq(loose.candidate_id)].sort_values(['m_B_GeV','m_ZD_GeV','ctau_mm'])
    if len(first)!=60 or not first['sample'].str.startswith('4Mu_').all() or not first['sample'].is_unique:
        raise AssertionError('Require exactly 60 separate matching 4Mu hypotheses.')
    order=first[['sample','m_B_GeV','m_ZD_GeV','ctau_mm']].rename(columns={
        'm_B_GeV':'parent_mass_GeV','m_ZD_GeV':'daughter_mass_GeV'}).reset_index(drop=True)
    order.insert(0,'model_index',finalize_4mu_final_09style_scans_np.arange(1,61))
    frames=[]
    for point,study in ((loose,'loose'),(gap,'gap')):
        part=signals.loc[signals.candidate_id.eq(point.candidate_id)].copy()
        if set(part['sample'])!=set(order['sample']):
            raise AssertionError('The two comparisons require identical separate signal models.')
        for metric in finalize_4mu_final_09style_scans_METRICS:
            finalize_4mu_final_09style_scans_np.testing.assert_allclose(part[metric].median(),point[metric+'_median'],rtol=1e-12,atol=1e-12)
        part=part.merge(order[['sample','model_index','parent_mass_GeV','daughter_mass_GeV']],
                        on='sample',validate='one_to_one')
        part['approach']=study
        frames.append(part)
    return finalize_4mu_final_09style_scans_pd.concat(frames,ignore_index=True),order


def finalize_4mu_final_09style_scans__meeting_comparison(result,loose,gap):
    data=result['tables']['data_closure']
    def closure(point,state,mass='intermediate'):
        part=data.loc[data.candidate_id.eq(point.candidate_id)&data.scope.eq(mass+'_'+state)]
        if len(part)!=1 or not bool(part.iloc[0].display_allowed):return 'DATA withheld'
        row=part.iloc[0]
        return f'{row.kappa:.3f} ± {row.kappa_error:.3f}' if finalize_4mu_final_09style_scans_np.isfinite(row.kappa) else 'Undefined'
    values=[['A definition',f'I<{loose.a_iso:g}, |Δφ|≥{loose.a_phi:g}',f'I<{gap.a_iso:.2f}, |Δφ|≥{gap.a_phi:g}'],
        ['Sideband boundaries',f'Ordinary I={loose.a_iso:g}, φ={loose.a_phi:g}',f'B/D: I≥{gap.b_iso:.2f}; C/D: φ<{gap.c_phi:g}'],
        ['Absolute median S/√(S+B)',f'{loose.s_sqrt_s_plus_b_median:.4f}',f'{gap.s_sqrt_s_plus_b_median:.4f}'],
        ['Absolute median known-background Z',f'{loose.z_known_b_median:.4f}',f'{gap.z_known_b_median:.4f}'],
        ['Absolute median finite-MC Z',f'{loose.z_mc_stat_diagnostic_median:.4f}',f'{gap.z_mc_stat_diagnostic_median:.4f}'],
        ['Maximum nominal BCD S/B_MC',f'{100*loose.max_S_B_BCD:.3f}%',f'{100*gap.max_S_B_BCD:.3f}%'],
        ['Nominal DATA B / C / D','DATA withheld','DATA withheld'],
        ['Uncorrected nominal B×C/D ± stat','Not evaluated','Not evaluated'],
        ['Intermediate PP DATA κ',closure(loose,'PP'),closure(gap,'PP')],
        ['Intermediate PD DATA κ',closure(loose,'PD'),closure(gap,'PD')],
        ['Intermediate DP DATA κ',closure(loose,'DP'),closure(gap,'DP')],
        ['Mass-inclusive PP DATA κ',closure(loose,'PP','all'),closure(gap,'PP','all')],
        ['Mass-inclusive PD DATA κ',closure(loose,'PD','all'),closure(gap,'PD','all')]]
    fig,axes=finalize_4mu_final_09style_scans__canvas('Final comparison: extended loose A and fixed-A buffer, neither selected',cols=1,height=8.4,bottom=.26,top=.87)
    ax=axes[0,0];ax.axis('off')
    table=ax.table(cellText=values,colLabels=['Evidence','Extended loose A (gate fails)','Fixed WP2 A + buffer comparison'],
        cellLoc='center',colWidths=[.40,.30,.30],bbox=[0,.08,1,.90]);finalize_4mu_final_09style_scans__style_table(table,9.6)
    fig.text(.075,.215,'No nominal B/C/D contamination-passing candidate exists. The loose MC-A maximum and fixed-WP2 buffer remain diagnostic comparisons.\n'
        'Loose expands A; buffer retains WP2 A. Buffer controls overlap extended loose A, so nominal DATA comparison is unavailable.\n'
        'Only explicitly released VR results appear. No closure correction, transfer nuisance or final statistical-model result is adopted.',
        fontsize=10.2,color='#25354A',va='top')
    finalize_4mu_final_09style_scans__caption(fig,'Sensitivity entries are absolute medians over the same 60 matching 4Mu hypotheses. Contamination uses 120 separately normalized hypotheses.\n'
        'The DATA rows retain their own integrated gate and public-reuse audit. Withheld values are not zero, including intermediate PP at the buffer comparison.\n'
        'A-based significance omits the complete control-count and closure likelihood. Nominal DATA A remains blinded; there is no new DATA query.')
    return fig


def finalize_4mu_final_09style_scans__unavailable_data_significance():
    fig,axes=finalize_4mu_final_09style_scans__canvas('DATA-sideband-based expected significance: not evaluated',cols=3,height=8.1,bottom=.34,top=.82)
    titles=[r'$S/\sqrt{S+P}$','Asimov Z: fixed ABCD estimate','Z: ABCD counting uncertainty']
    for ax,title in zip(axes[0],titles):
        ax.set_title(title,pad=25,fontsize=12.5)
        ax.set(xlim=(.5,60.5),xticks=[1,10,20,30,40,50,60],xlabel='Signal hypothesis index',yticks=[])
        ax.text(.5,.5,'Not evaluated\n\nNo nominal B/C/D gate pass\nDATA controls unavailable',
                transform=ax.transAxes,ha='center',va='center',fontsize=11,color='#666666')
        ax.text(.5,1.015,'Medians: unavailable',transform=ax.transAxes,fontsize=8.5,va='bottom',ha='center')
    finalize_4mu_final_09style_scans__caption(fig,'This preserves the original notebook-09 three-panel slot without inventing a DATA-based background or expected significance. No values are plotted.\n'
        'The raw prediction P=N_B×N_C/N_D requires approved nominal B/C/D counts. All notebook-10 candidates fail the existing all-120 central MC contamination gate.\n'
        'Nominal B/C/D counts and P therefore remain unavailable. They are not replaced by zero, direct MC A, or a different channel.\n'
        'The MC-A diagnostics appear on the preceding comparison page. A full ABCD likelihood with leakage and closure/transfer uncertainty has not been evaluated.\n'
        'Nominal DATA A remains blinded, and this presentation makes no new DATA query or disclosure.')
    return fig


# ========================================================================
# abcd_4mu_final_09style_validation: original study API
# ========================================================================
"""Notebook-09 renderers adapted to unchanged notebook-10 4mu measurements.

The MC and public-DATA tables are passed through, never recalculated or opened
from original event files. Red markers are diagnostic comparison points, not
adopted nominal candidates. Inclusive DATA remain explicit NaN placeholders.
"""
import hashlib as finalize_4mu_final_09style_validation_hashlib

import numpy as finalize_4mu_final_09style_validation_np
import pandas as finalize_4mu_final_09style_validation_pd
from matplotlib.legend_handler import HandlerTuple as finalize_4mu_final_09style_validation_HandlerTuple
from matplotlib.lines import Line2D as finalize_4mu_final_09style_validation_Line2D

finalize_4mu_final_09style_validation_style09 = _FINALIZE_NAMESPACES['abcd_iso_buffer_plots']
finalize_4mu_final_09style_validation__canvas = _FINALIZE_NAMESPACES['abcd_final_check_plots']._canvas; finalize_4mu_final_09style_validation__caption = _FINALIZE_NAMESPACES['abcd_final_check_plots']._caption
finalize_4mu_final_09style_validation__intermediate_overlay = _FINALIZE_NAMESPACES['abcd_final_comparison_plots']._intermediate_overlay


finalize_4mu_final_09style_validation_MASSES = ('low', 'intermediate', 'all')
finalize_4mu_final_09style_validation_STATES = ('PP', 'PD', 'DP')
finalize_4mu_final_09style_validation_MASS_LABELS = {
    'low': 'Low mass: 0 <= MJJ < 150 GeV',
    'intermediate': 'Intermediate mass: 150 <= MJJ < 300 GeV',
    'all': 'Mass inclusive: MJJ >= 0 GeV',
}
finalize_4mu_final_09style_validation_SCOPE_PREFIX = {'low': 'VR1', 'intermediate': 'VR2', 'all': 'INCLUSIVE'}
finalize_4mu_final_09style_validation_DATA_FIELDS = ['A', 'B', 'C', 'D', 'pred', 'pred_error', 'kappa', 'kappa_error',
               'A_error', 'B_error', 'C_error', 'D_error', 'var_A', 'var_B', 'var_C', 'var_D']


def finalize_4mu_final_09style_validation_make_validation_figures(result):
    """Return the 31 matching 09 slots using only supplied notebook-10 results."""
    signatures = {name: finalize_4mu_final_09style_validation__signature(result['tables'][name])
                  for name in ('mc_scan', 'data_closure', 'data_bins')}
    proxy, one_dimensional, audit = finalize_4mu_final_09style_validation__render_adapter(result)
    selections = proxy['selections']
    figures, captions, definitions = [], {}, []
    def register(name, figure, section, question, caption, *, source=None, mass=None):
        finalize_4mu_final_09style_validation__finish_figure(figure, caption, source=source, mass=mass)
        figures.append((name, figure))
        captions[name] = caption
        definitions.append(dict(name=name, section=section, question=question))
    for mass in finalize_4mu_final_09style_validation_MASSES:
        for source in ('MC', 'DATA'):
            key = 'if_vr_'+source.lower()+'_'+mass
            caption = finalize_4mu_final_09style_validation__scan_caption('loose', mass, source, selections)
            register(key, finalize_4mu_final_09style_validation_style09._vr_scan(proxy, mass, source), 'loose',
                     'How does '+source+' VR closure vary with isolation along the loose boundary?',
                     caption, source=source, mass=mass)
        key = 'loose_final_1d_'+mass
        register(key, finalize_4mu_final_09style_validation_style09._final_1d(proxy, 'loose', mass), 'loose',
                 'What are the Data yield, closure and contamination near the loose comparison point with extended A?',
                 finalize_4mu_final_09style_validation__final_caption('loose', mass, selections), source='DATA', mass=mass)
        for source in ('MC', 'DATA'):
            key = 'loose2d_'+source.lower()+'_'+mass
            register(key, finalize_4mu_final_09style_validation__two_dimensional(proxy, 'loose', mass, source), 'loose',
                     'What are the '+source+' 2D distribution and public mask for the loose boundary with extended A?',
                     finalize_4mu_final_09style_validation__plane_caption('loose', mass, source, selections), source=source, mass=mass)
        for source in ('MC', 'DATA'):
            key = 'gap_vr_'+source.lower()+'_'+mass
            figure = finalize_4mu_final_09style_validation_style09._gap_vr_scan(proxy, mass, source)
            register(key, figure, 'buffer',
                     'Is '+source+' VR closure stable against the angle edge of the fixed-A buffer?',
                     finalize_4mu_final_09style_validation__scan_caption('gap', mass, source, selections), source=source, mass=mass)
        key = 'buffer_final_1d_'+mass
        register(key, finalize_4mu_final_09style_validation_style09._final_1d(proxy, 'gap', mass), 'buffer',
                 'How do the buffer Data yield, closure and contamination compare along the same angle scan?',
                 finalize_4mu_final_09style_validation__final_caption('gap', mass, selections), source='DATA', mass=mass)
        for source in ('MC', 'DATA'):
            key = 'buffer2d_'+source.lower()+'_'+mass
            register(key, finalize_4mu_final_09style_validation__two_dimensional(proxy, 'gap', mass, source), 'buffer',
                     'What is the '+source+' 2D distribution for the WP2 fixed-A buffer?',
                     finalize_4mu_final_09style_validation__plane_caption('gap', mass, source, selections), source=source, mass=mass)
    overlay = one_dimensional.loc[one_dimensional.source.eq('DATA')
        & one_dimensional.mass_slice.eq('intermediate')
        & ((one_dimensional.study.eq('loose') & one_dimensional.scan_axis.eq('a_iso'))
           | (one_dimensional.study.eq('gap') & one_dimensional.scan_axis.eq('b_iso')))].copy()
    loose, gap = selections['loose'], selections['gap']
    caption = (
        'Intermediate MJJ 150-300 GeV; columns are PP/PD/DP, first leading and second subleading mu-LJ. '
        f'Loose varies I_A at phi_A={loose["a_phi"]:g}; buffer varies I_H with A fixed at '
        f'I_A<{gap["a_iso"]:g}, phi_A>={gap["a_phi"]:g} and phi_L={gap["c_phi"]:g}.\n'
        'Only existing same-scope public DATA are shown. Top: observed A and B*C/D; middle: kappa; '
        'bottom: maximum ABCD S/B_MC over 120 separate hypotheses. Neighboring cuts and both approaches share events.\n'
        f'The loose I_A={loose["a_iso"]:g} and buffer I_H={gap["b_iso"]:g} markers are separate diagnostic '
        'comparisons, not adopted nominal choices: no nominal B/C/D MC gate passes. '
        'The green 0.9-1.1 band is a visual guide, not a confidence interval. Nominal DATA remain unqueried.')
    register('final_intermediate_overlay', finalize_4mu_final_09style_validation__diagnostic_overlay(overlay, selections),
             'comparison', 'How do loose and buffer yields, closure and contamination differ in the same intermediate Data?', caption,
             source='DATA', mass='intermediate')
    if len(figures) != 31 or len({name for name, _ in figures}) != 31:
        raise AssertionError('The notebook-09 validation layout requires exactly 31 unique slots.')
    if signatures != {name: finalize_4mu_final_09style_validation__signature(result['tables'][name]) for name in signatures}:
        raise AssertionError('Rendering may not change any MC measurement or DATA value/gate.')
    validation = finalize_4mu_final_09style_validation_pd.DataFrame(audit)
    if not validation.passed.all():
        raise AssertionError('The 09 render adapter did not preserve the existing values.')
    return dict(figures=figures, captions=captions, definitions=finalize_4mu_final_09style_validation_pd.DataFrame(definitions),
        tables={'09style_validation_1d': one_dimensional,
                '09style_intermediate_overlay': overlay,
                '09style_validation_candidates': finalize_4mu_final_09style_validation_pd.DataFrame(
                    [dict(study=study, **spec) for study, spec in selections.items()]),
                '09style_validation_checks': validation},
        metadata=dict(source_tables_unchanged=True, original_DATA_files_opened=0,
            original_MC_files_opened=0, new_DATA_queries=0, source09_renderers_reused=True,
            diagnostic_comparison_only=True, adopted_nominal_candidate=None,
            inclusive_DATA='NO_PUBLIC_SAME_SELECTION; all values remain NaN',
            displacement_order='leading mu-LJ, subleading mu-LJ',
            mass_order=list(finalize_4mu_final_09style_validation_MASSES), displacement_columns=list(finalize_4mu_final_09style_validation_STATES), figures=31))


def finalize_4mu_final_09style_validation__signature(frame):
    digest = finalize_4mu_final_09style_validation_hashlib.sha256()
    digest.update('|'.join(frame.columns).encode())
    digest.update(finalize_4mu_final_09style_validation_pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def finalize_4mu_final_09style_validation__bool(series):
    return series.fillna(False).astype(str).str.lower().isin(('true', '1'))


def finalize_4mu_final_09style_validation__alias_scope(scope):
    if scope == 'nominal_DD':
        return 'Nominal_DD'
    mass, state = scope.rsplit('_', 1)
    return finalize_4mu_final_09style_validation_SCOPE_PREFIX[mass]+'_'+state


def finalize_4mu_final_09style_validation__render_adapter(result):
    display_mc_plane = _FINALIZE_NAMESPACES['abcd_4mu_final_09style_scans'].display_mc_plane
    scan = result['tables']['mc_scan'].copy()
    selections = finalize_4mu_final_09style_validation__comparison_specifications(result)
    loose_spec, gap_spec = selections['loose'], selections['gap']
    data = result['tables']['data_closure'].copy()
    bins = result['tables']['data_bins'].copy()
    checks = []
    assert not data.loc[data.mass_slice.eq('all'), finalize_4mu_final_09style_validation_DATA_FIELDS].notna().any().any()
    assert not finalize_4mu_final_09style_validation__bool(data.loc[data.mass_slice.eq('all'), 'display_allowed']).any()
    finalize_4mu_final_09style_validation__validate_data_source(data, bins)
    mc = scan.copy()
    mc['approved'] = True
    mc['display_allowed'] = True
    mc['reason'] = 'MC_ALWAYS_VISIBLE'
    mc['max_S_B'] = mc.max_S_B_ABCD
    mc['source'] = 'MC'
    data['signal_hypotheses'] = 120
    combined = finalize_4mu_final_09style_validation_pd.concat([mc, data], ignore_index=True, sort=False)
    combined['scope_original'] = combined.scope
    combined['scope'] = combined.scope.map(finalize_4mu_final_09style_validation__alias_scope)
    loose = combined.loc[combined.strategy.eq('loose')].copy()
    paths = []
    for study, axis, fixed, value, selected in (
            ('loose', 'a_iso', 'a_phi', loose_spec['a_phi'], loose_spec['a_iso']),
            ('loose', 'a_phi', 'a_iso', loose_spec['a_iso'], loose_spec['a_phi']),
            ('gap', 'c_phi', 'b_iso', gap_spec['b_iso'], gap_spec['c_phi']),
            ('gap', 'b_iso', 'c_phi', gap_spec['c_phi'], gap_spec['b_iso'])):
        if study == 'loose':
            frame = combined.loc[combined.strategy.eq('loose') & finalize_4mu_final_09style_validation_np.isclose(combined[fixed], value)].copy()
        else:
            frame = combined.loc[combined.strategy.eq(gap_spec['strategy'])
                                 & (combined.is_main.eq(True) | combined.is_main.isna())
                                 & finalize_4mu_final_09style_validation_np.isclose(combined[fixed], value)].copy()
        if axis in ('a_iso', 'b_iso'):
            frame = frame.loc[frame[axis].between(.2, .5)].copy()
        frame['study'] = study
        frame['scan_axis'] = axis
        frame['scan_value'] = frame[axis]
        frame['fixed_axis'] = fixed
        frame['fixed_value'] = value
        frame['selected_point'] = finalize_4mu_final_09style_validation_np.isclose(frame[axis], selected)
        paths.append(frame)
    one_dimensional = finalize_4mu_final_09style_validation_pd.concat(paths, ignore_index=True, sort=False)
    # This compatibility selector is private to rendering; it makes no physical selection.
    proxy = dict(tables={'if_scan': scan.loc[scan.scope.eq('nominal_DD') & scan.strategy.eq('loose')].copy(),
        'if_selection': finalize_4mu_final_09style_validation_pd.DataFrame([dict(role='global_operational_best', candidate_id=loose_spec['candidate_id']),
                                     dict(role='global_best', candidate_id=loose_spec['candidate_id'])]),
        'if_vr_closure': loose, 'vd_closure1d': one_dimensional},
        selections=selections, selected_planes={'loose': {}, 'gap': {}})
    for scope, native in result['backgrounds'].items():
        if scope == 'intermediate_DD':
            continue
        folded = display_mc_plane(native)
        shape = folded['MC'].shape
        part = bins.loc[bins.scope.eq(scope)].sort_values(['ix', 'iy'])
        if len(part):
            if len(part) != finalize_4mu_final_09style_validation_np.prod(shape):
                raise AssertionError('Existing public 2D bins must match the original09 display shape.')
            allowed = finalize_4mu_final_09style_validation__bool(part.display_allowed).to_numpy().reshape(shape)
            values = part.data.to_numpy(float).reshape(shape)
            variances = part.var_data.to_numpy(float).reshape(shape)
            maxima = part.max_S_B.to_numpy(float).reshape(shape)
            finalize_4mu_final_09style_validation_np.testing.assert_allclose(folded['MC'], part.background.to_numpy(float).reshape(shape),
                                       rtol=1e-10, atol=1e-9)
            finalize_4mu_final_09style_validation_np.testing.assert_allclose(folded['variance_MC'], part.var_background.to_numpy(float).reshape(shape),
                                       rtol=1e-10, atol=1e-9)
            reason = part.reason.to_numpy(object).reshape(shape)
        else:
            allowed = finalize_4mu_final_09style_validation_np.zeros(shape, bool)
            values = finalize_4mu_final_09style_validation_np.full(shape, finalize_4mu_final_09style_validation_np.nan)
            variances = finalize_4mu_final_09style_validation_np.full(shape, finalize_4mu_final_09style_validation_np.nan)
            maxima = finalize_4mu_final_09style_validation_np.full(shape, finalize_4mu_final_09style_validation_np.nan)
            reason = finalize_4mu_final_09style_validation_np.full(shape, 'NO_PUBLIC_SAME_SELECTION', object)
        if not finalize_4mu_final_09style_validation_np.isnan(values[~allowed]).all() or not finalize_4mu_final_09style_validation_np.isnan(variances[~allowed]).all():
            raise AssertionError('A hidden DATA bin must never become a visible zero.')
        for study, geometry in proxy['selections'].items():
            selected = data.loc[data.scope.eq(scope) & data.candidate_id.eq(geometry['candidate_id'])]
            if len(selected) != 1:
                raise AssertionError('Each diagnostic geometry needs one explicit same-scope DATA status.')
            gate = bool(finalize_4mu_final_09style_validation__bool(selected.display_allowed).iloc[0])
            proxy['selected_planes'][study][finalize_4mu_final_09style_validation__alias_scope(scope)] = dict(folded,
                DATA=values.copy(), variance_DATA=variances.copy(), display_allowed=allowed.copy(),
                reason=reason.copy(), max_S_B=maxima.copy(), integrated_allowed=gate,
                closure_gate=gate, closure_gate_reason=selected.reason.iloc[0],
                release_provenance=finalize_4mu_final_09style_validation_np.full(shape, 'EXACT_PUBLIC_ROW_SPACE_REUSE', object))
        checks.append(dict(scope=scope,
            check='MC folding matches existing 154-bin moments; DATA values and masks copied without new disclosure',
            passed=True))
    checks.extend([
        dict(scope='all', check='Inclusive DATA 1D and 2D are unavailable NaN placeholders, not zero', passed=True),
        dict(scope='all', check='The existing DATA measurement and display gates are unchanged', passed=True),
        dict(scope='Nominal_DD', check='No nominal DATA value enters any validation renderer', passed=True),
    ])
    return proxy, one_dimensional, checks


def finalize_4mu_final_09style_validation__validate_data_source(data, bins):
    hidden = data.loc[~finalize_4mu_final_09style_validation__bool(data.display_allowed)]
    if not hidden[finalize_4mu_final_09style_validation_DATA_FIELDS].isna().all().all():
        raise AssertionError('Masked integrated DATA must remain NaN.')
    if not bins.loc[~finalize_4mu_final_09style_validation__bool(bins.display_allowed), ['data', 'var_data']].isna().all().all():
        raise AssertionError('Masked DATA bins must remain NaN.')
    shown = bins.loc[finalize_4mu_final_09style_validation__bool(bins.display_allowed)]
    if not finalize_4mu_final_09style_validation_np.array_equal(shown.data.to_numpy(), shown.var_data.to_numpy()):
        raise AssertionError('Public DATA counts retain Poisson variance.')
    if (shown.max_S_B >= .1).any() or (shown.background <= 1e-8).any():
        raise AssertionError('Visible bins must retain the established MC contamination gate.')


def finalize_4mu_final_09style_validation__two_dimensional(proxy, study, mass, source):
    scopes = [finalize_4mu_final_09style_validation_SCOPE_PREFIX[mass]+'_'+state for state in finalize_4mu_final_09style_validation_STATES]
    planes = proxy['selected_planes'][study]
    label = 'Loose A-extended diagnostic' if study == 'loose' else 'Fixed WP2 A + diagnostic buffer'
    figure, axes = finalize_4mu_final_09style_validation__canvas(label+' · '+source+' 2D · '+finalize_4mu_final_09style_validation_MASS_LABELS[mass],
                          cols=3, height=8.9, bottom=.29, top=.83)
    figure.subplots_adjust(right=.93, wspace=.39)
    norm, exists = finalize_4mu_final_09style_validation_style09._norm([planes[scope][source] for scope in scopes])
    for ax, state, scope in zip(axes[0], finalize_4mu_final_09style_validation_STATES, scopes):
        plane = planes[scope]
        image = finalize_4mu_final_09style_validation_style09._plane_image(ax, plane, source, norm, proxy['selections'][study], title=state)
        colorbar = figure.colorbar(image, ax=ax, fraction=.040, pad=.020,
            label='MC weighted events / bin' if source == 'MC' else 'DATA events / bin')
        values = finalize_4mu_final_09style_validation_np.asarray(plane[source], float)
        if not exists or not finalize_4mu_final_09style_validation_np.any(finalize_4mu_final_09style_validation_np.isfinite(values) & (values > 0)):
            colorbar.ax.set_visible(False)
        ax.set_xlabel('Leading mu-LJ isolation')
    finalize_4mu_final_09style_validation_style09._distribution_legend(figure, data=source == 'DATA', gap=study == 'gap', bottom=.15)
    return figure


def finalize_4mu_final_09style_validation__diagnostic_text(text):
    replacements = (
        ('2μ2e', '4μ'), ('Selected loose optimum', 'Loose A-extended diagnostic (not adopted)'),
        ('Fixed WP2 A + selected buffer', 'Fixed WP2 A + diagnostic buffer'),
        ('Selected final comparison', 'Diagnostic comparison (not adopted)'),
        ('Selected buffer', 'WP2 diagnostic buffer (not adopted)'),
        ('Selected integrated DATA withheld', 'Comparison DATA withheld'),
        ('selected DATA withheld', 'comparison DATA withheld'),
        ('Selected DATA', 'Comparison DATA'), ('Selected MC', 'Comparison MC'),
        ('Selected A=', 'Comparison A='),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def finalize_4mu_final_09style_validation__finish_figure(figure, caption, source=None, mass=None):
    for text in list(figure.texts):
        if text.get_position()[1] < .05:
            text.remove()
        else:
            text.set_text(finalize_4mu_final_09style_validation__diagnostic_text(text.get_text()))
    if figure._suptitle is not None:
        title = figure._suptitle.get_text()
        title = title.replace('Loose A-extended diagnostic (not adopted)', 'Loose A-extended comparison')
        title = title.replace('Fixed WP2 A + diagnostic buffer', 'Fixed WP2 A + buffer comparison')
        title = title.replace('· final DATA closure ·', '· DATA closure ·')
        figure._suptitle.set_text(title)
    for ax in figure.axes:
        for text in ax.texts:
            text.set_text(finalize_4mu_final_09style_validation__diagnostic_text(text.get_text()))
        if 'Muon-LJ isolation' in ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel().replace('Muon-LJ isolation', 'Leading mu-LJ isolation'))
    for legend in figure.legends:
        for text in legend.texts:
            text.set_text(finalize_4mu_final_09style_validation__diagnostic_text(text.get_text()))
    if source == 'DATA' and mass == 'all':
        for ax in figure.axes:
            if ax.get_label() == '<colorbar>':
                continue
            # MC contamination may remain in the bottom row; only unavailable DATA panels get this label.
            label = ax.get_ylabel()
            if 'S/B' not in label and ('κ' in label or 'kappa' in label or 'DATA' in label
                                        or '|Δφ|' in label or ax.get_xlabel() == 'Leading mu-LJ isolation'):
                for text in list(ax.texts):
                    if text.get_text() == 'Integrated DATA withheld':
                        text.remove()
                ax.text(.5, .5, 'DATA unavailable\nNo existing public same-selection observations',
                    transform=ax.transAxes, ha='center', va='center', fontsize=10,
                    color='#777777', bbox=dict(facecolor='white', edgecolor='none', alpha=.9))
    finalize_4mu_final_09style_validation__caption(figure, caption)


def finalize_4mu_final_09style_validation__scan_caption(study, mass, source, selections):
    spec = selections[study]
    angles = ','.join(f'{angle:g}' for angle in finalize_4mu_final_09style_validation_style09.ANGLES)
    geometry = (f'Loose isolation scan at each native phi_A={angles}; red comparison: '
                f'I_A={spec["a_iso"]:g}, phi_A={spec["a_phi"]:g}. '
                'This A-extended diagnostic has the largest known-background MC-A median Z in the native loose scan before the contamination gate.'
                if study == 'loose' else
                f'Fixed A: I_A<{spec["a_iso"]:g} and phi_A>={spec["a_phi"]:g}; '
                f'I_H={spec["b_iso"]:g}; scan C/D angle upper edge phi_L; red comparison phi_L={spec["c_phi"]:g}.')
    availability = ('No inclusive same-selection public DATA exist: all DATA yields, predictions and kappa remain NaN. '
                    'MC contamination is not an inclusive DATA measurement.' if source == 'DATA' and mass == 'all' else
                    'DATA use only unchanged same-scope public observations and existing gates; withheld points are not zero.'
                    if source == 'DATA' else 'Only background MC closure is shown; MC weighted moments and errors are unchanged.')
    return (source+' only; '+finalize_4mu_final_09style_validation_MASS_LABELS[mass]+'. PP/PD/DP columns retain leading/subleading mu-LJ ordering.\n'
        +geometry+'\n'+availability+'\n'
        'The red point is diagnostic, not an adopted nominal choice: no nominal B/C/D gate passes. '
        'The 0.9-1.1 green band is a visual guide, not a confidence interval; neighboring cuts share events. '
        'Nominal DATA A/B/C/D remain unqueried.')


def finalize_4mu_final_09style_validation__final_caption(study, mass, selections):
    spec = selections[study]
    path = (f'vary loose I_A at phi_A={spec["a_phi"]:g}; comparison I_A={spec["a_iso"]:g}'
            if study == 'loose' else
            f'hold A at I_A<{spec["a_iso"]:g}, phi_A>={spec["a_phi"]:g} and '
            f'I_H={spec["b_iso"]:g}; vary phi_L; comparison phi_L={spec["c_phi"]:g}')
    return ('Existing public DATA only; '+finalize_4mu_final_09style_validation_MASS_LABELS[mass]+'. Columns: PP/PD/DP; '+path+'.\n'
        'Rows: observed A and raw B*C/D; kappa with propagated counting errors; maximum ABCD S/B_MC over 120 separate hypotheses. '
        'The bottom MC diagnostic does not authorize a DATA display. Hidden DATA are NaN, never zero.\n'
        +('There are no inclusive public observations with this same selection, so the first two rows are unavailable placeholders.\n'
           if mass == 'all' else '')+
        'Red marks the supplied diagnostic comparison, not an adopted working point. The green 0.9-1.1 band is a visual guide, not a confidence interval. '
        'Neighboring cuts share events; partial 2D displays do not replace integrated yields. Nominal DATA remain unqueried.')


def finalize_4mu_final_09style_validation__plane_caption(study, mass, source, selections):
    spec = selections[study]
    geometry = (f'A-extended loose: I_A=I_H={spec["a_iso"]:g}, phi_A=phi_L={spec["a_phi"]:g}.'
                if study == 'loose' else
                f'Fixed WP2 A: I_A<{spec["a_iso"]:g}, phi_A>={spec["a_phi"]:g}; '
                f'B/D I_H={spec["b_iso"]:g}; C/D phi_L={spec["c_phi"]:g}.')
    return (source+' only; '+finalize_4mu_final_09style_validation_MASS_LABELS[mass]+'. PP/PD/DP are leading/subleading mu-LJ displacement categories. '+geometry+'\n'
        'The geometry is a diagnostic comparison, not an adopted nominal definition. Colours are events per display bin with one shared source-specific logarithmic scale. '
        'DATA zeros are white; grey cross-hatching is unavailable DATA; amber hatching is an unused buffer.\n'
        'The final isolation display bin [1,2] contains all I>=1 tails and overflow; angle flows are folded into the first/last display bins; isolation underflow is excluded.\n'
        +('No inclusive same-selection public DATA exist; the three DATA planes remain all-NaN placeholders.'
           if source == 'DATA' and mass == 'all' else
           'All displayed DATA values/masks are copied from the already-public notebook-10 table; no original DATA is opened.'
           if source == 'DATA' else 'Only the unchanged original MC projection is folded for display; no DATA plane is selected.'))


def finalize_4mu_final_09style_validation__comparison_specifications(result):
    provided = finalize_4mu_final_09style_validation_pd.DataFrame(result['plot_candidates'])
    nominal = result['tables']['mc_scan']
    nominal = nominal.loc[nominal.scope.eq('nominal_DD')]
    specifications = {}
    for study, group in (('loose', 'loose_extended'), ('gap', 'buffer_wp2')):
        part = provided.loc[provided.group.eq(group)]
        if len(part) != 1:
            raise AssertionError('Exactly one explicitly supplied '+group+' diagnostic is required.')
        row = part.iloc[0]
        measured = nominal.loc[nominal.candidate_id.eq(row.candidate_id)]
        if len(measured) != 1:
            raise AssertionError('The supplied comparison must match one existing MC candidate.')
        measured = measured.iloc[0]
        if bool(measured.purity_pass_BCD) or row.get('role', 'diagnostic') in ('selected', 'adopted'):
            raise AssertionError('This renderer retains diagnostic comparisons only, with no adopted nominal candidate.')
        specification = {field: measured[field] for field in
                         ('candidate_id', 'strategy', 'a_iso', 'a_phi', 'b_iso', 'c_phi')}
        for field in ('a_iso', 'a_phi', 'b_iso', 'c_phi'):
            if field in row and not finalize_4mu_final_09style_validation_np.isclose(float(row[field]), float(measured[field])):
                raise AssertionError('Comparison geometry must equal the recorded MC candidate.')
        specification['group'] = group
        if study == 'loose':
            ordinary = nominal.loc[nominal.strategy.eq('loose')]
            if measured.strategy != 'loose' or not finalize_4mu_final_09style_validation_np.isclose(
                    measured.z_known_b_median, ordinary.z_known_b_median.max()):
                raise AssertionError('The supplied A-extended point must be the native loose MC-A maximum before gating.')
        specifications[study] = specification
    return specifications


def finalize_4mu_final_09style_validation__diagnostic_overlay(rows, selections):
    """Retain 09's 3x3 canvas while distinguishing the two isolation edges."""
    figure = finalize_4mu_final_09style_validation__intermediate_overlay(rows, selections)
    colors = {'loose': '#0072B2', 'gap': '#D55E00'}
    edges = {'loose': selections['loose']['a_iso'], 'gap': selections['gap']['b_iso']}
    for ax in figure.axes:
        old = [line for line in ax.lines if line.get_linestyle() == ':'
               and line.get_color() == '#444444'
               and finalize_4mu_final_09style_validation_np.asarray(line.get_xdata()).shape == (2,)
               and finalize_4mu_final_09style_validation_np.allclose(line.get_xdata(), edges['loose'])]
        if len(old) != 1:
            raise AssertionError('The original09 overlay must have exactly one comparison edge per panel.')
        old[0].remove()
        for study in ('loose', 'gap'):
            ax.axvline(edges[study], color=colors[study], ls=':', lw=1.2)
    legend = figure.legends[0]
    handles = list(legend.legend_handles)
    labels = [text.get_text() for text in legend.texts]
    marker_index = [i for i, label in enumerate(labels) if label.startswith('Selected isolation edge')]
    if len(marker_index) != 1 or len(handles) != 6:
        raise AssertionError('Retain the original09 six-entry, three-column comparison legend.')
    i = marker_index[0]
    handles[i] = tuple(finalize_4mu_final_09style_validation_Line2D([], [], color=colors[study], ls=':', lw=1.2)
                       for study in ('loose', 'gap'))
    labels[i] = f'Edges: loose I_A={edges["loose"]:g}; buffer I_H={edges["gap"]:g}'
    legend.remove()
    figure.legend(handles=handles, labels=labels, handler_map={tuple: finalize_4mu_final_09style_validation_HandlerTuple(ndivide=None)},
                  loc='lower center', bbox_to_anchor=(.5, .125), ncol=3,
                  frameon=False, fontsize=9)
    return figure


# ========================================================================
# abcd_data_expected_significance: original study API
# ========================================================================
"""Expected significance using signal MC A and already released DATA B/C/D.

No original DATA file or nominal DATA A is read. The raw ABCD prediction and
its counting variance are evaluated as background-only sideband diagnostics,
without a closure correction or a signal-leakage subtraction.
"""
import numpy as finalize_data_expected_significance_np
import pandas as finalize_data_expected_significance_pd
from matplotlib.lines import Line2D as finalize_data_expected_significance_Line2D

finalize_data_expected_significance__canvas = _FINALIZE_NAMESPACES['abcd_final_check_plots']._canvas; finalize_data_expected_significance__caption = _FINALIZE_NAMESPACES['abcd_final_check_plots']._caption
finalize_data_expected_significance_COLORS = _FINALIZE_NAMESPACES['abcd_final_comparison_plots'].COLORS; finalize_data_expected_significance_LABELS = _FINALIZE_NAMESPACES['abcd_final_comparison_plots'].LABELS; finalize_data_expected_significance_final_comparison_tables = _FINALIZE_NAMESPACES['abcd_final_comparison_plots'].final_comparison_tables
finalize_data_expected_significance_asimov_significance = _FINALIZE_NAMESPACES['review_mc_sr_working_points'].asimov_significance


finalize_data_expected_significance_METRICS = ('s_sqrt_s_plus_p', 'z_data_b_fixed', 'z_data_b_counting')


def finalize_data_expected_significance_data_expected_significance_definitions():
    return finalize_data_expected_significance_pd.DataFrame([
        ('final_data_expected_significance', 'DATA-sideband-based expected significance',
         'Compare signal MC A with raw nominal DATA B*C/D, including a separate ABCD counting-uncertainty diagnostic.')],
        columns=['name', 'title', 'purpose'])


def finalize_data_expected_significance_data_expected_significance_figures(result):
    tables = finalize_data_expected_significance_data_expected_significance_tables(result)
    return [(finalize_data_expected_significance__data_expected_plot(tables['data_expected_signals'],
                                tables['data_expected_summary'], result['selections']),
             'final_data_expected_significance')]


def finalize_data_expected_significance_data_expected_significance_tables(result):
    """Return per-model values, two control summaries and independent checks."""
    tables = result['tables']
    comparison = tables.get('final_comparison_signals')
    if comparison is None:
        comparison = finalize_data_expected_significance_final_comparison_tables(result)['final_comparison_signals']
    controls = tables.get('nominal_controls', tables.get('sbd_summary'))
    if controls is None:
        raise AssertionError('Already released nominal B/C/D controls are required.')
    gate = tables['nominal_final_gate']
    contamination = tables['if_nominal_data_contamination']
    signal_frames, summaries, validation = [], [], []
    for study in ['loose', 'gap']:
        spec = result['selections'][study]
        identity = spec['candidate_id']
        control = controls.loc[controls.candidate_id.eq(identity)]
        if len(control) != 1:
            raise AssertionError('Require one exact final nominal-control definition: '+identity)
        control = control.iloc[0]
        if not bool(control.display_allowed) or bool(control.nominal_A_evaluated):
            raise AssertionError('Use only approved B/C/D controls with nominal A untouched.')
        for field in ['a_iso', 'a_phi', 'b_iso', 'c_phi']:
            if not finalize_data_expected_significance_np.isclose(control[field], spec[field]):
                raise AssertionError('Control counts must have the exact final candidate geometry.')
        decision = gate.loc[gate.candidate_id.eq(identity)]
        purity = contamination.loc[contamination.candidate_id.eq(identity)]
        if (len(decision) != 3 or set(decision.region) != set('BCD')
                or not decision.purity_pass.all() or decision.overlap_final_A.any()
                or decision.nominal_A_evaluated.any()):
            raise AssertionError('The final B/C/D gate must pass without any overlap with final A.')
        if (len(purity) != 3 or set(purity.region) != set('BCD')
                or not purity.signal_hypotheses.eq(120).all()
                or not purity.mc_purity_pass.all() or not purity.display_allowed.all()
                or not purity.max_S_B_MC.lt(.1).all() or purity.nominal_A_evaluated.any()):
            raise AssertionError('Every final control must pass the established all-120 MC contamination gate.')
        counts = finalize_data_expected_significance_np.array([control['data_'+region] for region in 'BCD'], float)
        variances = finalize_data_expected_significance_np.array([control['var_'+region] for region in 'BCD'], float)
        if not finalize_data_expected_significance_np.isfinite(counts).all() or (counts <= 0).any() or not finalize_data_expected_significance_np.array_equal(counts, finalize_data_expected_significance_np.rint(counts)):
            raise AssertionError('The current Poisson counting diagnostic requires positive integer controls.')
        finalize_data_expected_significance_np.testing.assert_array_equal(variances, counts)
        nb, nc, nd = counts
        prediction = nb*nc/nd
        variance = (nc/nd)**2*variances[0]+(nb/nd)**2*variances[1]+(nb*nc/nd**2)**2*variances[2]
        finalize_data_expected_significance_np.testing.assert_allclose(variance, prediction**2*finalize_data_expected_significance_np.sum(1/counts), rtol=1e-13)
        finalize_data_expected_significance_np.testing.assert_allclose([prediction, finalize_data_expected_significance_np.sqrt(variance)],
                                   [control.pred, control.pred_error], rtol=1e-12, atol=1e-12)
        part = comparison.loc[comparison.approach.eq(study)].sort_values('model_index').copy()
        if len(part) != 60 or not part['sample'].is_unique or not part.final_candidate_id.eq(identity).all():
            raise AssertionError('Require the same 60 final signal-MC hypotheses per approach.')
        source_ids = part.source_A_candidate_id.unique()
        if len(source_ids) != 1:
            raise AssertionError('Each final A must use one nominal signal-MC source coordinate.')
        source_point = tables['if_scan'].loc[tables['if_scan'].candidate_id.eq(source_ids[0])]
        if (len(source_point) != 1 or not finalize_data_expected_significance_np.isclose(source_point.iloc[0].a_iso, spec['a_iso'])
                or not finalize_data_expected_significance_np.isclose(source_point.iloc[0].a_phi, spec['a_phi'])):
            raise AssertionError('Signal-MC A must match the exact final A boundaries.')
        original = tables['if_signals'].loc[tables['if_signals'].candidate_id.eq(source_ids[0])]
        original = original.set_index('sample').loc[part['sample']]
        finalize_data_expected_significance_np.testing.assert_allclose(part[['S_A', 'var_S_A']], original[['S_A', 'var_S_A']],
                                   rtol=1e-12, atol=1e-12)
        signal = part.S_A.to_numpy(float)
        if not finalize_data_expected_significance_np.isfinite(signal).all() or (signal <= 0).any():
            raise AssertionError('The absolute logarithmic comparison requires positive signal MC yields.')
        keep = ['approach', 'final_candidate_id', 'source_A_candidate_id', 'sample',
                'model_index', 'parent_mass_GeV', 'daughter_mass_GeV', 'ctau_mm', 'S_A', 'var_S_A']
        part = part[keep].copy()
        part['data_B'], part['data_C'], part['data_D'] = counts
        part['background_data_abcd'] = prediction
        part['variance_background_counting'] = variance
        part['background_counting_error'] = finalize_data_expected_significance_np.sqrt(variance)
        part['s_sqrt_s_plus_p'] = signal/finalize_data_expected_significance_np.sqrt(signal+prediction)
        part['z_data_b_fixed'] = finalize_data_expected_significance_asimov_significance(signal, prediction)
        part['z_data_b_counting'] = finalize_data_expected_significance_asimov_significance(signal, prediction, variance)
        if not finalize_data_expected_significance_np.isfinite(part[list(finalize_data_expected_significance_METRICS)].to_numpy(float)).all():
            raise AssertionError('All DATA-sideband-based expected metrics must be finite.')
        if (part.z_data_b_counting > part.z_data_b_fixed+1e-12).any():
            raise AssertionError('Counting uncertainty must not increase the expected significance.')
        # Independent quadrature of the Asimov log-likelihood derivative checks
        # the reused stable implementation, including very small s/b values.
        for field, test_variance in [('z_data_b_fixed', 0.), ('z_data_b_counting', variance)]:
            numerical = finalize_data_expected_significance__quadrature_expected_z(signal, prediction, test_variance)
            finalize_data_expected_significance_np.testing.assert_allclose(part[field], numerical, rtol=2e-10, atol=2e-12)
        part['nominal_A_evaluated'] = False
        part['closure_correction_applied'] = False
        part['control_signal_subtraction_applied'] = False
        part['new_data_evaluated'] = False
        signal_frames.append(part)
        summary = dict(approach=study, candidate_id=identity, data_B=nb, data_C=nc, data_D=nd,
            background_data_abcd=prediction, variance_background_counting=variance,
            background_counting_error=finalize_data_expected_significance_np.sqrt(variance), counting_relative_error=finalize_data_expected_significance_np.sqrt(variance)/prediction,
            signal_hypotheses=60, max_BCD_S_B_MC=float(purity.max_S_B_MC.max()),
            nominal_A_evaluated=False, new_data_evaluated=False,
            closure_correction_applied=False, control_signal_subtraction_applied=False,
            interpretation='Expected diagnostic conditional on raw background-only sidebands; no observed A and no full ABCD likelihood')
        summary.update({field+'_median':float(part[field].median()) for field in finalize_data_expected_significance_METRICS})
        summaries.append(summary)
        validation.extend([
            dict(scope=identity, check='Exact released final B/C/D geometry; all-120 MC contamination pass; final A untouched', passed=True),
            dict(scope=identity, check='Raw B*C/D and independently propagated Poisson variance reproduce stored prediction and error', passed=True),
            dict(scope=identity, check='All 60 fixed-background and counting-uncertainty values agree with independent likelihood-derivative quadrature', passed=True),
            dict(scope=identity, check='Counting uncertainty does not increase Z; no closure correction, control signal subtraction or new DATA query', passed=True),
        ])
    combined = finalize_data_expected_significance_pd.concat(signal_frames, ignore_index=True)
    first = combined.loc[combined.approach.eq('loose'), ['model_index', 'sample']].reset_index(drop=True)
    second = combined.loc[combined.approach.eq('gap'), ['model_index', 'sample']].reset_index(drop=True)
    finalize_data_expected_significance_pd.testing.assert_frame_equal(first, second)
    validation.append(dict(scope='Final DATA-based comparison',
        check='Both approaches use exactly the same 60 separately normalized signal hypotheses and index ordering', passed=True))
    return dict(data_expected_signals=combined, data_expected_summary=finalize_data_expected_significance_pd.DataFrame(summaries),
                data_expected_validation=finalize_data_expected_significance_pd.DataFrame(validation))


def finalize_data_expected_significance__quadrature_expected_z(signal, background, variance):
    nodes, weights = finalize_data_expected_significance_np.polynomial.legendre.leggauss(64)
    ratio = finalize_data_expected_significance_np.asarray(signal, dtype=finalize_data_expected_significance_np.longdouble)/finalize_data_expected_significance_np.longdouble(background)
    alpha = finalize_data_expected_significance_np.longdouble(variance)/(finalize_data_expected_significance_np.longdouble(background)+finalize_data_expected_significance_np.longdouble(variance))
    x = ratio[:, None]*(finalize_data_expected_significance_np.asarray(nodes, dtype=finalize_data_expected_significance_np.longdouble)[None, :]+1)/2
    integrand = finalize_data_expected_significance_np.log1p(x)-finalize_data_expected_significance_np.log1p(alpha*x)
    integral = ratio/2*finalize_data_expected_significance_np.sum(integrand*finalize_data_expected_significance_np.asarray(weights, dtype=finalize_data_expected_significance_np.longdouble)[None, :], axis=1)
    return finalize_data_expected_significance_np.asarray(finalize_data_expected_significance_np.sqrt(2*finalize_data_expected_significance_np.longdouble(background)*integral), dtype=float)


def finalize_data_expected_significance__data_expected_plot(rows, summary, selections):
    fig, axes = finalize_data_expected_significance__canvas('DATA-sideband-based expected significance: signal MC A + DATA B×C/D',
                        cols=3, height=8.1, bottom=.34, top=.82)
    titles = [r'$S/\sqrt{S+P}$', 'Asimov Z: fixed ABCD estimate',
              'Z: ABCD counting uncertainty']
    values = rows[list(finalize_data_expected_significance_METRICS)].to_numpy(float)
    if (values <= 0).any():
        raise AssertionError('Require positive expected metrics for the logarithmic display.')
    limits = (float(values.min())*.55, float(values.max())*2)
    parents = rows.loc[rows.approach.eq('loose')].groupby('parent_mass_GeV', sort=True)
    for ax, metric, title in zip(axes[0], finalize_data_expected_significance_METRICS, titles):
        for number, (mass, part) in enumerate(parents):
            left, right = part.model_index.min()-.5, part.model_index.max()+.5
            if number % 2 == 0:
                ax.axvspan(left, right, color='#E8EFF8', alpha=.5, zorder=0)
            ax.text((left+right)/2, .98, f'{mass:g} GeV', ha='center', va='top',
                    transform=ax.get_xaxis_transform(), fontsize=8, color='#444444')
            if number:
                ax.axvline(left, color='#8795A7', lw=.9)
            for edge in [left+5, left+10]:
                ax.axvline(edge, color='#B6C0CB', ls=':', lw=.7)
        for study, marker in [('loose', 'o'), ('gap', '^')]:
            part = rows.loc[rows.approach.eq(study)].sort_values('model_index')
            ax.plot(part.model_index, part[metric], marker=marker, ls='none', color=finalize_data_expected_significance_COLORS[study],
                    ms=3.4, alpha=.85, label=finalize_data_expected_significance_LABELS[study])
        ax.set(ylabel='Absolute '+('S / √(S+P)' if metric=='s_sqrt_s_plus_p' else 'expected Z'),
               yscale='log', ylim=limits, xlim=(.5,60.5), xticks=[1,10,20,30,40,50,60],
               xlabel='Signal hypothesis index')
        ax.set_title(title, pad=25, fontsize=12.5)
        ax.tick_params(axis='x', labelsize=8)
        medians = [rows.loc[rows.approach.eq(s),metric].median() for s in ['loose','gap']]
        ax.text(.5,1.015,f'Medians: loose {medians[0]:.5f}; buffer {medians[1]:.5f}',
                transform=ax.transAxes,fontsize=8.5,va='bottom',ha='center')
    indexed = summary.set_index('approach')
    handles = []
    for study, marker in [('loose','o'),('gap','^')]:
        item = indexed.loc[study]
        handles.append(finalize_data_expected_significance_Line2D([],[],color=finalize_data_expected_significance_COLORS[study],marker=marker,ls='none',
            label=f'{finalize_data_expected_significance_LABELS[study]}: P={item.background_data_abcd:.2f} ± {item.background_counting_error:.2f} (stat)'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.225),ncol=2,frameon=False,fontsize=9.5)
    finalize_data_expected_significance__caption(fig,'S is the signal-MC yield in each final A. P=N_B×N_C/N_D uses already released nominal DATA B/C/D; no nominal DATA A, total or projection is evaluated.\n'
             'The first two panels hold P fixed. The last uses V_P=P²(1/N_B+1/N_C+1/N_D) in an effective auxiliary-count Asimov diagnostic; it is not a full three-control Poisson likelihood.\n'
             'The raw prediction assumes background-only sidebands: no signal-leakage subtraction or closure correction is applied. Closure/transfer and signal-MC uncertainties are not included.\n'
             'These are conditional expected-sensitivity diagnostics, not observed significances or final expected-limit results. No reference-WP normalization is applied.\n'
             'The same 60 separate signal models and index ordering are used as in the MC-background comparison; the complete mapping and values are exported in data_expected_signals.csv.')
    return fig
