"""Module to define miscellaneous helper methods"""

import yaml
import numpy as np
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import hist.intervals
from sidm import BASE_DIR

def print_list(l):
    """Print one list element per line"""
    print('\n'.join(l))

def print_debug(name, val, print_mode=True):
    """Print variable name and value"""
    if print_mode:
        print(f"{name}: {val}")

def partition_list(l, condition):
    """Given a single list, return separate lists of elements that pass or fail a condition"""
    passes = []
    fails = []
    for x in l:
        if condition(x):
            passes.append(x)
        else:
            fails.append(x)
    return passes, fails

def flatten(x):
    """Flatten arbitrarily nested list or dict"""
    # https://stackoverflow.com/questions/2158395/
    flattened_list = []
    def loop(sublist):
        if isinstance(sublist, dict):
            sublist = sublist.values()
        for item in sublist:
            if isinstance(item, (dict, list)):
                loop(item)
            else:
                flattened_list.append(item)
    loop(x)
    return flattened_list

def add_unique_and_flatten(flattened_list, x):
    """Flatten arbitrarily nested list or dict, keeping only unique items"""
    # https://stackoverflow.com/questions/2158395/
    def loop(sublist):
        if isinstance(sublist, dict):
            sublist = sublist.values()
        for item in sublist:
            if isinstance(item, (dict, list)):
                loop(item)
            elif item not in flattened_list:
                flattened_list.append(item)
    loop(x)
    return flattened_list

def as_int(array):
    """Return array with values converted to ints"""
    return ak.values_astype(array, "int64")

def dR(obj1, obj2):
    """Return dR between obj1 and the nearest obj2; returns inf if no obj2 is found"""
    dr = obj1.nearest(obj2, return_metric=True)[1]
    return ak.fill_none(dr, np.inf)

def dR_general(obj1, obj2):
    """Return ΔR between obj1 and obj2, filling None with inf"""
    return ak.fill_none(obj1.delta_r(obj2), np.inf)

def dR_outer(obj1, obj2):
    """Return dR between outer tracks of obj1 and obj2"""
    return ak.fill_none(np.sqrt((obj1.outerEta - obj2.outerEta)**2 + (obj1.outerPhi - obj2.outerPhi)**2), np.inf)

def drop_none(obj):
    """Remove None entries from an array (not available in Awkward 1)"""
    return obj[~ak.is_none(obj, axis=1)] # fixme: not clear why axis=1 works and axis=-1 doesn't

def matched(obj1, obj2, r):
    """Return set of obj1 that have >=1 obj2 within r; remove None entries before returning"""
    return drop_none(obj1[dR(obj1, obj2) < r])

def add_matched_dsamuon_mass(obj):
    obj["mass"] = ak.full_like(obj.pt, 0.105712890625)
    return obj

def lj_combination_dR(obj):
    pair = ak.combinations(obj, 2, axis=1, fields=["lj1", "lj2"])
    dR = dR_general(pair["lj1"], pair["lj2"])
    min_dR = ak.min(dR_general(pair["lj1"], pair["lj2"]), axis=1)
    max_dR = ak.max(dR_general(pair["lj1"], pair["lj2"]), axis=1)
    return dR, min_dR, max_dR

def pick_leptonlike_pdgid(obj):
    mask_e = (abs(obj.pdgId) == 11)
    mask_mu = (abs(obj.pdgId) == 13)
    mask_pho = (abs(obj.pdgId) == 22)

    e, mu, pho = obj[mask_e], obj[mask_mu], obj[mask_pho]
    return e, mu, pho

def pick_e_mother_category(obj):
    e_mask = (abs(obj.pdgId) == 11)
    obj = obj[e_mask]
    mother = obj.distinctParent.pdgId
    
    mask_dp = (abs(mother) == 32)
    mask_W = (abs(mother) == 24)
    mask_DB = (abs(mother) == 411) | (abs(mother) == 421) | (abs(mother) == 423) | (abs(mother) == 431) | (abs(mother) == 511) | (abs(mother) == 521) | (abs(mother) == 531) | (abs(mother) == 541)
    mask_pion = (abs(mother) == 111) | (abs(mother) == 211)

    m_dp, m_W, m_DB, m_pion = obj[mask_dp], obj[mask_W], obj[mask_DB], obj[mask_pion]
    return obj, m_dp, m_W, m_DB, m_pion

def pick_mu_mother_category(obj):
    mu_mask = (abs(obj.pdgId) == 13)
    obj = obj[mu_mask]
    mother = abs(obj.distinctParent.pdgId)
    
    mask_dp = (abs(mother) == 32)
    mask_Z = (abs(mother) == 23)
    mask_D = (abs(mother) == 411) | (abs(mother) == 421) | (abs(mother) == 423) | (abs(mother) == 431)
    mask_B = (abs(mother) == 511) | (abs(mother) == 521) | (abs(mother) == 531) | (abs(mother) == 541)
    mask_qg = (abs(mother) == 1) | (abs(mother) == 2) | (abs(mother) == 3) | (abs(mother) == 4) | (abs(mother) == 5) | (abs(mother) == 6) | (abs(mother) == 9) | (abs(mother) == 21)
    mask_pion = (abs(mother) == 111) | (abs(mother) == 211)
    
    m_dp, m_Z, m_D, m_B, m_qg, m_pion = obj[mask_dp], obj[mask_Z], obj[mask_D], obj[mask_B], obj[mask_qg], obj[mask_pion]
    return obj, m_dp, m_Z, m_D, m_B, m_qg, m_pion

def pick_pho_mother_category(obj):
    pho_mask = (abs(obj.pdgId) == 22)
    obj = obj[pho_mask]
    mother = obj.distinctParent.pdgId
    
    mask_e = (abs(mother) == 11)
    mask_mu = (abs(mother) == 13)
    mask_Z = (abs(mother) == 23)
    mask_pion = (abs(mother) == 111) | (abs(mother) == 211)
    mask_qg = (abs(mother) == 1) | (abs(mother) == 2) | (abs(mother) == 3) | (abs(mother) == 4) | (abs(mother) == 5) | (abs(mother) == 6) | (abs(mother) == 9) | (abs(mother) == 21)
    
    m_e, m_mu, m_Z, m_pion, m_qg = obj[mask_e], obj[mask_mu], obj[mask_Z], obj[mask_pion], obj[mask_qg]
    return obj, m_e, m_mu, m_Z, m_pion, m_qg

def pick_all_mother_category(obj):
    all_mask = (abs(obj.pdgId) == 11) | (abs(obj.pdgId) == 13) | (abs(obj.pdgId) == 22)
    obj = obj[all_mask]
    mother = obj.distinctParent.pdgId

    mask_dp = (abs(mother) == 32)
    mask_W = (abs(mother) == 24)
    mask_Z = (abs(mother) == 23)
    mask_DB = (abs(mother) == 411) | (abs(mother) == 421) | (abs(mother) == 423) | (abs(mother) == 431) | (abs(mother) == 511) | (abs(mother) == 521) | (abs(mother) == 531) | (abs(mother) == 541)
    mask_qg = (abs(mother) == 1) | (abs(mother) == 2) | (abs(mother) == 3) | (abs(mother) == 4) | (abs(mother) == 5) | (abs(mother) == 6) | (abs(mother) == 9) | (abs(mother) == 21)
    mask_pion = (abs(mother) == 111) | (abs(mother) == 211)
    mask_e = (abs(mother) == 11)
    mask_mu = (abs(mother) == 13)
    
    m_dp, m_W, m_Z, m_DB, m_qg, m_pi, m_e, m_mu = obj[mask_dp], obj[mask_W], obj[mask_Z], obj[mask_DB], obj[mask_qg], obj[mask_pion], obj[mask_e], obj[mask_mu]
    return obj, m_dp, m_W, m_Z, m_DB, m_qg, m_pi, m_e, m_mu

def rho(obj, ref=None, use_v=False):
    """Return transverse distance between object and reference (default reference is 0,0)"""
    if use_v:
        obj_x = obj.vx
        obj_y = obj.vy
        ref_x = ref.vx if ref is not None else 0.0
        ref_y = ref.vy if ref is not None else 0.0
    else:
        obj_x = obj.x
        obj_y = obj.y
        ref_x = ref.x if ref is not None else 0.0
        ref_y = ref.y if ref is not None else 0.0
    return np.sqrt((obj_x - ref_x)**2 + (obj_y - ref_y)**2)

def dxy(obj, ref=None):
    """Return transverse distance between obj and ref at their point of closest approach"""
    # caveats discussed here apply: https://github.com/cms-sw/cmssw/blob/1bd97a649226ce2c2585f8b61f210aab6d0d4c44/DataFormats/TrackReco/interface/TrackBase.h#L678-L683
    shape = ak.ones_like(obj.vx)
    x_val = ak.flatten(ref.x) if ref is not None else 0.0
    y_val = ak.flatten(ref.y) if ref is not None else 0.0
    ref_x = x_val*shape
    ref_y = y_val*shape
    return (-(obj.vx - ref_x)*obj.py + (obj.vy - ref_y)*obj.px)/obj.pt

def lxy(obj):
    """Return transverse distance between production and decay vertices"""
    return rho(obj, ak.firsts(obj.children, axis=2), use_v=True)

def set_plot_style(style='cms', dpi=50):
    """Set plotting style using mplhep"""
    if style == 'cms':
        plt.style.use(hep.style.CMS)
    else:
        raise NotImplementedError
    plt.rcParams['figure.dpi'] = dpi

def plot(hists, skip_label=False, **kwargs):
    """Plot using hep.hist(2d)plot and add cms labels"""

    # set default arguments
    default_kwargs = {
        'flow': "sum",
    }
    kwargs = {**default_kwargs, **kwargs}

    dim = len(hists[0].axes) if isinstance(hists, list) else len(hists.axes)
    if dim == 1:
        h = hep.histplot(hists, **kwargs)
    elif dim == 2:
        h = hep.hist2dplot(hists, **kwargs)
    else:
        raise NotImplementedError(f"Cannot plot {dim}-dimensional hist")
    if not skip_label:
        hep.cms.label()
    return h

def get_eff_hist(num_hist, denom_hist):
    """Returns the histogram of num_hist/denom_hist and a 2D numpy array of the up/down errors on the efficiency. Plot the errors using yerr=errors when plotting. """
    denom_vals  = denom_hist.values()
    num_vals   = num_hist.values()

    errors = hist.intervals.ratio_uncertainty(num_vals,denom_vals,'efficiency')
    eff_values = num_vals/denom_vals

    eff_hist = hist.Hist(*num_hist.axes)
    eff_hist.values()[:] = eff_values
    return eff_hist, errors

def load_yaml(cfg):
    """Load yaml files and return corresponding dict"""
    with open(cfg, encoding="utf8") as yaml_cfg:
        return yaml.safe_load(yaml_cfg)

def make_fileset(samples, ntuple_version, max_files=-1, location_cfg="signal_v8.yaml", fileset=None):
    """Make fileset to pass to processor.runner"""
    # assume location_cfg is stored in sidm/configs/ntuples/
    location_cfg = f"{BASE_DIR}/configs/ntuples/" + location_cfg
    locations = load_yaml(location_cfg)[ntuple_version]
    if not fileset:
        fileset = {}
    for sample in samples:
        sample_yaml = locations["samples"][sample]
        base_path = locations["path"] + sample_yaml["path"]
        file_list = [base_path + f for f in sample_yaml["files"]]
        if max_files != -1:
            file_list = file_list[:max_files]
        fileset[sample] = {
            "files": file_list,
            "metadata": {
                "skim_factor": sample_yaml.get("skim_factor", 1.0)}
        }
    return fileset

def check_bit(array, bit_num):
    """Return boolean stored in the bit_numth bit of array"""
    return (array & pow(2, bit_num)) > 0

def check_bits(array, bit_nums):
    result = (array & pow(2, bit_nums[0]))>0
    for x in bit_nums[1:]:
        result = (result & ((array & pow(2, x))>0))>0
    return result

def get_hist_mean(h):
    """Return mean of 1D histogram"""
    return np.atleast_1d(h.profile(axis=0).view())[0].value

def plot_ratio(num, den, **kwargs):
    plt.subplots(2, 1, figsize=(10, 10), sharex=True,
                      gridspec_kw={'height_ratios': [2, 1],'hspace':0})
    plt.subplot(2, 1, 1)
    plot(num, flow='none')
    plot(den, flow='none')
    if "legend" in kwargs:
        plt.legend(kwargs["legend"])
    plt.subplot(2, 1, 2)
    eff, errors = get_eff_hist(num, den)
    plot(eff,yerr=errors,skip_label=True,color="black")
    plt.ylabel("Efficiency")
    plt.ylim(0, 1.2)

def round_sigfig(val, digits=1):
    """Return a number rounded to a given number of significant figures. Uses magic copied from
    https://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python"""
    return float('{:g}'.format(float('{:.{p}g}'.format(val, p=digits))))

def proper_ctau(bs, zd, lab_ct, grid_cfg=f"{BASE_DIR}/configs/signal_grid.yaml"):
    """Convert average lab-frame transverse decay length in cm to proper decay
    length in mm for SIDM signals"""
    grid = load_yaml(grid_cfg)
    # handle goofy edge cases that I suspect stems from Weinan rounding errors
    if (float(bs), float(zd), float(lab_ct)) == (150, 0.25, 150):
        proper_ct = 6.7
    elif (float(bs), float(zd), float(lab_ct)) == (150, 5, 150):
        proper_ct = 130.0
    elif (float(bs), float(zd), float(lab_ct)) == (800, 0.25, 150):
        proper_ct = 1.2
    else:
        proper_ct = lab_ct/grid[bs][zd]["labframe_factor"]
    return round_sigfig(proper_ct, digits=2)

def lab_ctau(bs, zd, proper_ct, grid_cfg=f"{BASE_DIR}/configs/signal_grid.yaml"):
    """Convert proper decay length in mm to average lab-frame transverse decay
    length in cm for SIDM signals"""
    grid = load_yaml(grid_cfg)
    # handle goofy edge case that I suspect stems from Weinan rounding errors
    if (float(bs), float(zd), float(proper_ct)) == (150, 0.25, 6.7):
        lab_ct = 150.0
    elif (float(bs), float(zd), float(proper_ct)) == (150, 5, 130):
        lab_ct = 150.0
    elif (float(bs), float(zd), float(proper_ct)) == (800, 0.25, 1.2):
        lab_ct = 150.0
    else:
        lab_ct = proper_ct*grid[bs][zd]["labframe_factor"]
    return round_sigfig(lab_ct, digits=2)

def get_xs(dataset, cfg="cross_sections.yaml"):
    """Fetch dataset xs from cfg"""
    # assume location_cfg is stored in sidm/configs/
    xs_menu = load_yaml(f"{BASE_DIR}/configs/" + cfg)
    try:
        return xs_menu[dataset]
    except KeyError:
        if dataset.startswith(("2Mu2E", "4Mu")):
            print("Signal not in xs cfg, assuming 1fb")
            return 0.001
        else:
            raise

def get_lumi(year, cfg="run_periods.yaml"):
    """Fetch run period lumi from cfg"""
    # assume location_cfg is stored in sidm/configs/
    lumi_menu = load_yaml(f"{BASE_DIR}/configs/" + cfg)
    return lumi_menu[year]["lumi"]

def get_lumixs_weight(dataset, year, sum_weights):
    """Get weights to scale n_evts to lumi*xs"""
    # n_evts: sum of weights from processed events
    lumi = get_lumi(year)
    xs = get_xs(dataset)
    return lumi*xs/sum_weights

def check_variablePhoton(value, min_val=0b01):
    """
    Function to check if the variable is at least `min_val`
    """
    return value >= min_val

def select_numbersPhoton(number, var1, var2):
    """
    Function to select the numbers where each variable is at least 0b010 except for variable of choice
    """
    selected = True
    
    # number will have 14 bits (2 bits per each cut)
    # starting with MinPtCut at the LSB
    # and ending with PhoIsoWithEALinScalingCut at the MSB
    variables = [
        ('MinPtCut', 0),  # 2 bits for MinPtCut (LSB)
        ('PhoSCEtaMultiRangeCut', 2),
        ('PhoSingleTowerHadOverEmCut', 4),
        ('PhoFull5x5SigmaIEtaIEtaCut', 6),
        ('ChHadIsoWithEALinScalingCut', 8),
        ('NeuHadIsoWithEAQuadScalingCut', 10),
        ('PhoIsoWithEALinScalingCut', 12),
    ]
    
    # Check each variable except variable of choice 
    for var, start_bit in variables:
        # Get the 2 bits corresponding to this variable
        value = (number >> start_bit) & 0b11  # Extract 2 bits
        if (var != f'{var1}') and (var != f'{var2}'):
            if not check_variablePhoton(value):
                selected = False
                break
    
    return selected


def returnBitMapTArrayPhoton(bitMap, var1, var2):
    tList = []
    for i in range(len(bitMap)):
        temp = []
        if len(bitMap[i]) == 0:
            tList.append(temp)
            continue
        for j in range(len(bitMap[i])):
            if select_numbersPhoton(bitMap[i][j], var1, var2):
                temp.append(True)
            else:
                temp.append(False)
        tList.append(temp)
    return ak.Array(tList)

def numClose(self, other, r):
    distance = other.metric_table(self)
    close = ak.count(distance[distance < r], axis=1)
    return close
