"""Utilities for blinded, counting-experiment ABCD limits with CMS Combine."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


REGIONS = ("A", "B", "C", "D")
EXPECTED_QUANTILES = {
    0.025: "exp_m2",
    0.160: "exp_m1",
    0.500: "exp_median",
    0.840: "exp_p1",
    0.975: "exp_p2",
}


def safe_name(value: str) -> str:
    """Return a filename/Combine-name-safe signal-point label."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def render_datacard(
    signal_name: str,
    sidebands: dict,
    signal: dict,
    kappa_relative_uncertainty: float = 0.0,
) -> str:
    """Build a four-channel counting datacard.

    B, C and D background yields are free rateParams constrained by their
    channel Poisson terms. The A rateParam is B*C/D. The observation in A is
    the blinded nominal prediction and is irrelevant for `-t -1` Asimov runs.
    """
    b, c, d = (float(sidebands[r]["yield"]) for r in ("B", "C", "D"))
    if min(b, c, d) <= 0:
        raise ValueError("B, C and D sideband yields must be positive")
    signal_yields = {r: float(signal[r]["yield"]) for r in REGIONS}
    if min(signal_yields.values()) < 0:
        raise ValueError("Combine counting cards cannot contain negative signal rates")

    a = b * c / d
    observation = {"A": a, "B": b, "C": c, "D": d}
    bins = [region for region in REGIONS for _ in (0, 1)]
    processes = [process for _ in REGIONS for process in ("signal", "background")]
    process_ids = [value for _ in REGIONS for value in (0, 1)]
    rates = [
        value
        for region in REGIONS
        for value in (signal_yields[region], 1.0)
    ]

    lines = [
        "imax 4",
        "jmax 1",
        "kmax *",
        "------------",
        "bin          " + " ".join(REGIONS),
        "observation  " + " ".join(f"{observation[r]:.10g}" for r in REGIONS),
        "------------",
        "bin          " + " ".join(f"{x:>12}" for x in bins),
        "process      " + " ".join(f"{x:>12}" for x in processes),
        "process      " + " ".join(f"{x:>12d}" for x in process_ids),
        "rate         " + " ".join(f"{x:>12.10g}" for x in rates),
        "------------",
        f"bkg_B rateParam B background {b:.10g} [1e-6,{max(10*b, 1e4):.10g}]",
        f"bkg_C rateParam C background {c:.10g} [1e-6,{max(10*c, 1e4):.10g}]",
        f"bkg_D rateParam D background {d:.10g} [1e-6,{max(10*d, 1e4):.10g}]",
        "bkg_A rateParam A background (@0*@1/@2) bkg_B,bkg_C,bkg_D",
    ]
    if kappa_relative_uncertainty > 0:
        factor = 1.0 + float(kappa_relative_uncertainty)
        # Columns follow A(sig,bkg), B(sig,bkg), C(sig,bkg), D(sig,bkg).
        lines.append(
            f"kappa_nonclosure lnN - {factor:.8g} - - - - - -"
        )
    lines.extend(
        [
            "------------",
            f"# signal_point {signal_name}",
            f"# blinded_A_prediction {a:.10g}",
            "# Run expected-only: combine -M AsymptoticLimits -d workspace.root -t -1 --expectSignal 0",
        ]
    )
    return "\n".join(lines) + "\n"


def write_datacard(
    output_dir: Path,
    signal_name: str,
    sidebands: dict,
    signal: dict,
    kappa_relative_uncertainty: float = 0.0,
) -> Path:
    """Write one deterministic datacard and return its path."""
    tag = f"{safe_name(signal_name)}_kappa{int(round(100*kappa_relative_uncertainty)):03d}"
    directory = Path(output_dir) / tag
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "datacard.txt"
    path.write_text(
        render_datacard(
            signal_name, sidebands, signal, kappa_relative_uncertainty
        )
    )
    return path


def combine_environment(cmssw_base: str | Path | None = None) -> dict:
    """Report whether direct or CMSSW-runtime Combine commands are available."""
    direct = {
        name: shutil.which(name)
        for name in ("combine", "text2workspace.py", "combineTool.py")
    }
    base = Path(cmssw_base).resolve() if cmssw_base else None
    return {
        **direct,
        "cmssw_base": str(base) if base else None,
        "cmssw_valid": bool(base and (base / "src").is_dir()),
    }


def _runtime_prefix(cmssw_base: str | Path | None) -> str:
    if cmssw_base is None:
        return ""
    base = Path(cmssw_base).resolve()
    if not (base / "src").is_dir():
        raise FileNotFoundError(f"Invalid CMSSW area: {base}")
    return (
        "source /cvmfs/cms.cern.ch/cmsset_default.sh >/dev/null 2>&1; "
        f"cd {base}/src; eval \"$(scram runtime -sh)\"; "
    )


def run_combine_card(
    datacard: Path,
    mass_label: int = 125,
    cmssw_base: str | Path | None = None,
    r_max: float = 1000,
    max_r_max: float = 1e8,
    timeout: int = 1800,
) -> dict:
    """Run expected AsymptoticLimits, increasing rMax for incomplete outputs."""
    datacard = Path(datacard).resolve()
    workdir = datacard.parent
    workspace = workdir / "workspace.root"
    tag = safe_name(workdir.name)
    prefix = _runtime_prefix(cmssw_base)
    root_file = workdir / f"higgsCombine.{tag}.AsymptoticLimits.mH{int(mass_label)}.root"
    attempts = []
    current_r_max = float(r_max)
    while True:
        command = (
            f"{prefix}"
            f"cd {workdir}; "
            f"text2workspace.py {datacard.name} -o {workspace.name}; "
            f"combine -M AsymptoticLimits -d {workspace.name} "
            f"-t -1 --expectSignal 0 --rMax {current_r_max:.10g} "
            f"-m {int(mass_label)} -n .{tag}"
        )
        result = subprocess.run(
            ["bash", "-lc", command],
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        attempts.append(current_r_max)
        limits_complete = False
        if result.returncode == 0 and root_file.is_file():
            try:
                read_asymptotic_limits(root_file)
                limits_complete = True
            except ValueError:
                pass
        if limits_complete or result.returncode != 0 or current_r_max >= max_r_max:
            break
        current_r_max = min(10 * current_r_max, float(max_r_max))

    return {
        "returncode": result.returncode,
        "command": command,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "workspace": str(workspace),
        "root_file": str(root_file),
        "root_exists": root_file.is_file(),
        "limits_complete": limits_complete,
        "r_max_attempts": attempts,
    }


def read_asymptotic_limits(root_file: str | Path) -> dict:
    """Read observed and expected quantiles from a Combine limit tree."""
    import uproot

    with uproot.open(root_file) as source:
        tree = source["limit"]
        limits = np.asarray(tree["limit"].array(library="np"), dtype=float)
        quantiles = np.asarray(
            tree["quantileExpected"].array(library="np"), dtype=float
        )
    result = {"observed": np.nan}
    for quantile, value in zip(quantiles, limits):
        if quantile < 0:
            result["observed"] = float(value)
            continue
        target = min(EXPECTED_QUANTILES, key=lambda x: abs(x - quantile))
        if abs(target - quantile) < 0.01:
            result[EXPECTED_QUANTILES[target]] = float(value)
    missing = set(EXPECTED_QUANTILES.values()) - result.keys()
    if missing:
        raise ValueError(f"Missing expected quantiles in {root_file}: {sorted(missing)}")
    return result


def collect_combine_results(manifest: pd.DataFrame) -> pd.DataFrame:
    """Parse every existing ROOT output named in a card manifest."""
    rows = []
    for record in manifest.to_dict("records"):
        root_file = Path(record["root_file"])
        if not root_file.is_file():
            continue
        rows.append({**record, **read_asymptotic_limits(root_file)})
    return pd.DataFrame(rows)


def plot_brazil(
    frame: pd.DataFrame,
    x: str,
    xlabel: str,
    ylabel: str = r"95% CL upper limit on signal strength $\mu$",
    title: str | None = None,
    include_observed: bool = False,
):
    """Draw a standard Combine Brazil-band plot from parsed quantiles."""
    import matplotlib.pyplot as plt

    required = {"exp_m2", "exp_m1", "exp_median", "exp_p1", "exp_p2", x}
    missing = required - set(frame)
    if missing:
        raise KeyError(f"Missing Brazil columns: {sorted(missing)}")
    ordered = frame.sort_values(x)
    xv = ordered[x].to_numpy(float)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.fill_between(
        xv, ordered.exp_m2, ordered.exp_p2, color="#F5D547",
        label=r"Expected $\pm2\sigma$",
    )
    ax.fill_between(
        xv, ordered.exp_m1, ordered.exp_p1, color="#55B748",
        label=r"Expected $\pm1\sigma$",
    )
    ax.plot(xv, ordered.exp_median, "k--", linewidth=2, label="Median expected")
    if include_observed and "observed" in ordered:
        ax.plot(xv, ordered.observed, "k-", linewidth=2, label="Observed")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax
