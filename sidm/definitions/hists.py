"""Define all available histograms

All hists are defined as Histogram objects whose axes are given as a list of Axis objects, which
bundle a hist.axis with a function that defines how the axis will be filled. The underlying
hist.Hists storage is weight unless otherwise specified.
"""

# python
import math
import importlib
# columnar analysis
import hist
import awkward as ak
# local
from sidm.tools import histogram as h
from sidm.tools.utilities import dR, lxy, matched
from sidm.definitions.objects import derived_objs
# always reload local modules to pick up changes during development
importlib.reload(h)


# define counters
counter_defs = {
    "Gen As to muons": lambda objs: ak.count(objs["genAs_toMu"].pt),
    "Gen As to electrons": lambda objs: ak.count(objs["genAs_toE"].pt),
    "Matched gen As to muons": lambda objs: ak.count(derived_objs["genAs_toMu_matched_lj"](objs, 0.4).pt),
    "Matched gen As to electrons": lambda objs: ak.count(derived_objs["genAs_toE_matched_lj"](objs, 0.4).pt),
    
    "Total LJs": lambda objs: ak.count(objs["ljs"].pt),
    "Mu-LJs": lambda objs: ak.count(objs["mu_ljs"].pt),
    "EGM-LJs": lambda objs: ak.count(objs["egm_ljs"].pt),
    
    "Total Matched-Jets": lambda objs: ak.count(derived_objs["matched_jets"](objs,0.4).pt),
    "Leading-Mu-Matched-Jets": lambda objs: ak.count(derived_objs["leading_mu_matched_jets"](objs,0.4).pt),
    "Subleading-Mu-Matched-Jets": lambda objs: ak.count(derived_objs["subleading_mu_matched_jets"](objs,0.4).pt),
    "EGM-Matched-Jets": lambda objs: ak.count(derived_objs["egm_matched_jets"](objs,0.4).pt),
    
}


# define default labels and binnings
obj_labels = {
    "electrons": "Electron",
    "photons": "Photon",
    "muons": "PF Muon",
    "dsaMuons": "DSA Muon",
    "ljs": "Lepton Jet",
    "mu_ljs": r"$\mu$-type Lepton Jet",
    "egm_ljs": r"$e\gamma$-type Lepton Jet",
    "electron_ljs": "Electron Lepton Jet",
    "photon_ljs": "Photon Lepton Jet",
    "pfmu_ljs": "PF Muon Lepton Jet",
    "dsamu_ljs": "DSA Muon Lepton Jet",
    "genAs": r"$Z_d$",
    "genAs_toMu": r"$Z_d\rightarrow \mu\mu$",
    "genAs_toE": r"$Z_d\rightarrow ee$",
    "pvs": "PV",
}
attr_labels = {
    "pt": r"$p_T$",
    "eta": r"$\eta$",
    "phi": r"$\phi$",
    "lxy": r"$L_{{xy}}$",
    "dxy": r"$d_0$",
}
default_binnings = {
    "n":  (10, 0, 10),
    "pt":  (50, 0, 100),
    "eta": (50, -3, 3),
    "phi": (50, -1*math.pi, math.pi),
    "lxy": (100, 0, 100),
}


# define convenience functions to simplify creating basic hists
def make_label(obj, attr, absval):
    obj_label = obj_labels.get(obj, obj)
    if attr == "n":
        return f"Number of {obj_label}s"
    attr = attr_labels.get(attr, attr)
    if absval:
        attr = f"|{attr}|"
    return f"{obj_label} {attr}"

def obj_attr(obj, attr, absval=False, nbins=None, xmin=None, xmax=None, label=None):
    (_nbins, _xmin, _xmax) = default_binnings.get(attr, (100, 0, 100))
    nbins = _nbins if nbins is None else nbins
    xmin = _xmin if xmin is None else xmin
    xmax = _xmax if xmax is None else xmax
    label = make_label(obj, attr, absval) if label is None else label
    return h.Histogram.simple_hist(obj, attr, absval, nbins, xmin, xmax, label)

def make_2d(h1, h2):
    return h.Histogram([h1.axes[-1], h2.axes[-1]])

def obj_eta_phi(obj, nbins_x=None, xmin=None, xmax=None, nbins_y=None, ymin=None, ymax=None):
    return make_2d(
        obj_attr(obj, "eta", nbins_x, xmin, xmax),
        obj_attr(obj, "phi", nbins_y, ymin, ymax),
    )


# define histograms
hist_defs = {
    # LJ-matched jet
    "lj_matchedjet_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 500, name="LJ_PT", label=r"LJ PT"),
                   lambda objs, mask: objs["ljs"][mask].pt),
            
            h.Axis(hist.axis.Regular(50, 0, 500, name="matched_jets_pt",
                                     label=r"Matched Jet PT"),
                   lambda objs, mask: derived_objs["matched_jets"](objs,0.4)[mask].pt),
        ],
        evt_mask=lambda objs: (ak.num(derived_objs["matched_jets"](objs,0.4)) == ak.num(objs["ljs"])),
    ),
    
    "lj_matchedjet_nmuon": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="Number of Muons in Matched-Jet"),
                   lambda objs, mask: derived_objs["matched_jets"](objs,0.4)[mask].nMuons),
        ],
    ),
    
    "lj_matchedjet_nelectron": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="Number of Electrons in Matched-Jet"),
                   lambda objs, mask: derived_objs["matched_jets"](objs,0.4)[mask].nElectrons),
        ],
    ),
    
    "leading_lj_matchedjet_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 500, name="LJ_PT", label=r"Leading LJ PT"),
                   lambda objs, mask: objs["ljs"][:,0:1][mask].pt),
            
            h.Axis(hist.axis.Regular(50, 0, 500, name="matched_jets_pt",
                                     label=r"Leading-Matched Jet PT"),
                   lambda objs, mask: derived_objs["leading_matched_jets"](objs,0.4)[mask].pt),
        ],
        evt_mask=lambda objs: (ak.num(derived_objs["leading_matched_jets"](objs,0.4)) == ak.num(objs["ljs"][:,0:1])),
    ),
    
    "subleading_lj_matchedjet_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 500, name="LJ_PT", label=r"Subleading LJ PT"),
                   lambda objs, mask: objs["ljs"][:,1:2][mask].pt),
            
            h.Axis(hist.axis.Regular(50, 0, 500, name="matched_jets_pt",
                                     label=r"Subleading-Matched Jet PT"),
                   lambda objs, mask: derived_objs["subleading_matched_jets"](objs,0.4)[mask].pt),
        ],
        evt_mask=lambda objs: (ak.num(derived_objs["subleading_matched_jets"](objs,0.4)) == ak.num(objs["ljs"][:,1:2])),
    ),
    
    "leading_mu_lj_matchedjet_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 500, name="LJ_PT", label=r"Leading Mu-LJ PT"),
                   lambda objs, mask: objs["mu_ljs"][:,0:1][mask].pt),
            
            h.Axis(hist.axis.Regular(50, 0, 500, name="matched_jets_pt",
                                     label=r"Leading-Mu-Matched Jet PT"),
                   lambda objs, mask: derived_objs["leading_mu_matched_jets"](objs,0.4)[mask].pt),
        ],
        evt_mask=lambda objs: (ak.num(derived_objs["leading_mu_matched_jets"](objs,0.4)) == ak.num(objs["mu_ljs"][:,0:1]))
    ),
    
    "leading_mu_lj_matchedjet_nmuon": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="Number of Muons in Leading Mu-Matched-Jet"),
                   lambda objs, mask: derived_objs["leading_mu_matched_jets"](objs,0.4)[mask].nMuons),
        ],
    ),
    
    "subleading_mu_lj_matchedjet_nmuon": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="Number of Muons in Subleading Mu-Matched-Jet"),
                   lambda objs, mask: derived_objs["subleading_mu_matched_jets"](objs,0.4)[mask].nMuons),
        ],
    ),
    
    "subleading_mu_lj_matchedjet_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 500, name="LJ_PT", label=r"Subleading Mu-LJ PT"),
                   lambda objs, mask: objs["mu_ljs"][:,1:2][mask].pt),
            
            h.Axis(hist.axis.Regular(50, 0, 500, name="matched_jets_pt",
                                     label=r"Subleading Mu-Matched Jet PT"),
                   lambda objs, mask: derived_objs["subleading_mu_matched_jets"](objs,0.4)[mask].pt),
        ],
        evt_mask=lambda objs: (ak.num(derived_objs["subleading_mu_matched_jets"](objs,0.4)) == ak.num(objs["mu_ljs"][:,1:2]))
    ),
    
    "egm_lj_matchedjet_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 500, name="LJ_PT", label=r"EGM-LJ PT"),
                   lambda objs, mask: objs["egm_ljs"][:,0:1][mask].pt),
            
            h.Axis(hist.axis.Regular(50, 0, 500, name="matched_jets_pt",
                                     label=r"EGM-Matched Jet PT"),
                   lambda objs, mask: derived_objs["egm_matched_jets"](objs,0.4)[mask].pt),
        ],
        evt_mask=lambda objs: (ak.num(derived_objs["egm_matched_jets"](objs,0.4)) == ak.num(objs["egm_ljs"][:,0:1])),
    ),
    
    "egm_lj_matchedjet_nelectron": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="Number of Electrons in EGM-Matched-Jet"),
                   lambda objs, mask: derived_objs["egm_matched_jets"](objs,0.4)[mask].nElectrons),
        ],
    ),
    
    # Matched Jet
    "matched_jets_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 500, name="matched_jets_pt", label="Matched Jet PT (|$\Delta$R| 0.4) [GeV]"),
                   lambda objs, mask: derived_objs["matched_jets"](objs,0.4)[mask].pt),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["matched_jets"](objs,0.4)) > 0,
    ),   
    
    "leading_matched_jets_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 500, name="leading_matched_jets_pt", label="Leading-Matched Jet PT (|$\Delta$R| 0.4) [GeV]"),
                   lambda objs, mask: derived_objs["leading_matched_jets"](objs,0.4)[mask].pt),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["leading_matched_jets"](objs,0.4)) > 0,
    ),
    
    "subleading_matched_jets_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 500, name="matched_subleadingjets_pt", label="Subleading-Matched Jet PT (|$\Delta$R| 0.4) [GeV]"),
                   lambda objs, mask: derived_objs["subleading_matched_jets"](objs,0.4)[mask].pt),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["subleading_matched_jets"](objs,0.4)) > 0,
    ),
    
    "leadingLJ_matchedjet_absdR": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 4, name="|$\Delta$R| (Leading LJ, Leading-Matched Jet)"),
                   lambda objs, mask: objs["ljs"][mask,0].delta_r(derived_objs["leading_matched_jets"](objs,0.4)[mask,0])),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["leading_matched_jets"](objs,0.4)) > 0,
    ),
    
    "subleadingLJ_matchedjet_absdR": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 4, name="|$\Delta$R| (Subleading LJ, Subleading-Matched Jet)"),
                   lambda objs, mask: objs["ljs"][mask,1].delta_r(derived_objs["subleading_matched_jets"](objs,0.4)[mask,0])),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["subleading_matched_jets"](objs,0.4)) > 0,
    ),
    
    "matchedjet_matchedjet_absdR": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 5, name="|$\Delta$R| (Leading-Matched Jet, Subleading-Matched Jet)"),
                   lambda objs, mask: derived_objs["leading_matched_jets"](objs,0.4)[mask,0].delta_r(derived_objs["subleading_matched_jets"](objs,0.4)[mask,0])),
        ],
        evt_mask=lambda objs: (ak.num(derived_objs["leading_matched_jets"](objs,0.4)) > 0) & (ak.num(derived_objs["subleading_matched_jets"](objs,0.4)) > 0),
    ), 
    
    "matchedjet_matchedjet_absdphi": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 2*math.pi, name="|$\Delta\phi$| (Leading-Matched Jet, Subleading-Matched Jet)"),
                   lambda objs, mask: abs(derived_objs["leading_matched_jets"](objs,0.4)[mask,0].phi - derived_objs["subleading_matched_jets"](objs,0.4)[mask,0].phi)),
        ],
        evt_mask=lambda objs: (ak.num(derived_objs["leading_matched_jets"](objs,0.4)) > 0) & (ak.num(derived_objs["subleading_matched_jets"](objs,0.4)) > 0),
    ), 
    
    "mu_matched_jets_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 500, name="mu_matched_jets_pt", label="Mu-LJ Matched Jet PT (|$\Delta$R| 0.4) [GeV]"),
                   lambda objs, mask: derived_objs["mu_matched_jets"](objs,0.4)[mask].pt),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["mu_matched_jets"](objs,0.4)) > 0,
    ),

    "leading_mu_matched_jets_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 500, name="leading_mu_matched_jets_pt", label="Leading-Mu-LJ Matched Jet PT (|$\Delta$R| 0.4) [GeV]"),
                   lambda objs, mask: derived_objs["leading_mu_matched_jets"](objs,0.4)[mask].pt),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["leading_mu_matched_jets"](objs,0.4)) > 0,
    ),

    "subleading_mu_matched_jets_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 500, name="subleading_mu_matched_jets_pt", label="Subleading-Mu-LJ Matched Jet PT (|$\Delta$R| 0.4) [GeV]"),
                   lambda objs, mask: derived_objs["subleading_mu_matched_jets"](objs,0.4)[mask].pt),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["subleading_mu_matched_jets"](objs,0.4)) > 0,
    ),   
    
    "egm_matched_jets_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 500, name="egm_matched_jets_pt", label="EGM-LJ Matched Jet PT (|$\Delta$R| 0.4) [GeV]"),
                   lambda objs, mask: derived_objs["egm_matched_jets"](objs,0.4)[mask].pt),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["egm_matched_jets"](objs,0.4)) > 0,
    ),   
    
    "all_egm_matched_jets_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 500, name="egm_matched_jets_pt", label="EGM-LJ Matched Jet PT (|$\Delta$R| 0.4) [GeV]"),
                   lambda objs, mask: derived_objs["all_egm_matched_jets"](objs,0.4)[mask].pt),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["all_egm_matched_jets"](objs,0.4)) > 0,
    ),   
    
    "mu_matchedjet_mu_matchedjet_absdR": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 5, name="|$\Delta$R| (Leading-Mu-LJ Matched Jet, Subleading-Mu-LJ Matched Jet)"),
                   lambda objs, mask: derived_objs["leading_mu_matched_jets"](objs,0.4)[mask,0].delta_r(derived_objs["subleading_mu_matched_jets"](objs,0.4)[mask,0])),
        ],
        evt_mask=lambda objs: (ak.num(derived_objs["leading_mu_matched_jets"](objs,0.4)) > 0) & (ak.num(derived_objs["subleading_mu_matched_jets"](objs,0.4)) > 0),
    ), 
    
    "mu_matchedjet_egm_matchedjet_absdR": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 5, name="|$\Delta$R| (Mu-LJ Matched Jet, EGM-LJ Matched Jet)"),
                   lambda objs, mask: derived_objs["leading_mu_matched_jets"](objs,0.4)[mask,0].delta_r(derived_objs["egm_matched_jets"](objs,0.4)[mask,0])),
        ],
        evt_mask=lambda objs: (ak.num(derived_objs["leading_mu_matched_jets"](objs,0.4)) > 0) & (ak.num(derived_objs["egm_matched_jets"](objs,0.4)) > 0),
    ), 

    "matched_jets_n": h.Histogram(
        [
            h.Axis(hist.axis.Regular(10, 0, 10, name="matched_jets_n", label="Number of Matched Jet (|$\Delta$R| 0.4)"),
                   lambda objs, mask: ak.num(derived_objs["matched_jets"](objs,0.4))),
        ],
    ),    
    
    "leading_matched_jets_n": h.Histogram(
        [
            h.Axis(hist.axis.Regular(10, 0, 10, name="leading_matched_jets_n", label="Number of Leading-Matched Jet (|$\Delta$R| 0.4)"),
                   lambda objs, mask: ak.num(derived_objs["leading_matched_jets"](objs,0.4))),
        ],
    ),   
    
    "subleading_matched_jets_n": h.Histogram(
        [
            h.Axis(hist.axis.Regular(10, 0, 10, name="subleading_matched_jets_n", label="Number of Subleading-Matched Jet (|$\Delta$R| 0.4)"),
                   lambda objs, mask: ak.num(derived_objs["subleading_matched_jets"](objs,0.4))),
        ],
    ),
    
    "leading_mu_matched_jets_n": h.Histogram(
        [
            h.Axis(hist.axis.Regular(10, 0, 10, name="leading_mu_matched_jets_n", label="Number of Leading-Mu-LJ Matched Jet (|$\Delta$R| 0.4)"),
                   lambda objs, mask: ak.num(derived_objs["leading_mu_matched_jets"](objs,0.4))),
        ],
    ),   
    
    "subleading_mu_matched_jets_n": h.Histogram(
        [
            h.Axis(hist.axis.Regular(10, 0, 10, name="subleading_mu_matched_jets_n", label="Number of Subleading-Mu-LJ Matched Jet (|$\Delta$R| 0.4)"),
                   lambda objs, mask: ak.num(derived_objs["subleading_mu_matched_jets"](objs,0.4))),
        ],
    ),   
    
    "egm_matched_jets_n": h.Histogram(
        [
            h.Axis(hist.axis.Regular(10, 0, 10, name="egm_matched_jets_n", label="Number of EGM-LJ Matched Jet (|$\Delta$R| 0.4)"),
                   lambda objs, mask: ak.num(derived_objs["egm_matched_jets"](objs,0.4))),
        ],
    ),   
    
    "matchedjet_matchedjet_invmass": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 1200, name="matchedjet_matchedjet_invmass",
                                     label=r"Invariant Mass (Matched Leading, Subleading Jets)"),
                   lambda objs, mask: derived_objs["matched_jets"](objs,0.4)[mask, :2].sum().mass),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["matched_jets"](objs,0.4)) > 1,
    ),
    
    "mu_matchedjet_egm_matchedjet_invmass": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 1200, name="matchedjet_invmass",
                                     label=r"Invariant Mass (Mu-LJ Matched Jets, EGM-LJ Matched Jets)"),
                   lambda objs, mask: (derived_objs["leading_mu_matched_jets"](objs,0.4)[mask,0] + derived_objs["egm_matched_jets"](objs,0.4)[mask,0]).mass),
        ],
        evt_mask=lambda objs: (ak.num(derived_objs["leading_mu_matched_jets"](objs,0.4)) > 0) & (ak.num(derived_objs["egm_matched_jets"](objs,0.4)) > 0),
    ),
    
    "mu_matchedjet_mu_matchedjet_invmass": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 1200, name="matchedjet_invmass",
                                     label=r"Invariant Mass (Leading, Subleading Mu-LJ Matched Jets)"),
                   lambda objs, mask: (derived_objs["leading_mu_matched_jets"](objs,0.4)[mask,0] + derived_objs["subleading_mu_matched_jets"](objs,0.4)[mask,0]).mass),
        ],
        evt_mask=lambda objs: (ak.num(derived_objs["leading_mu_matched_jets"](objs,0.4)) > 0) & (ak.num(derived_objs["subleading_mu_matched_jets"](objs,0.4)) > 0),
    ),
    
    "leading_matchedjet_lj_ptRatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 3.0, name="leading_matchedjet_lj_ptRatio",
                   label="Leading-Matched Jets PT / Leading LJ PT"),
                   lambda objs, mask: (derived_objs["leading_matched_jets"](objs,0.4)[mask,0].pt) / (objs["ljs"][:,0:1][mask].pt)),
        ],
        evt_mask=lambda objs: (ak.num(derived_objs["leading_matched_jets"](objs,0.4)) > 0) & (ak.num(objs["ljs"][:,0:1]) > 0),
    ),
    
    "subleading_matchedjet_lj_ptRatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 3.0, name="subleading_matchedjet_lj_ptRatio",
                   label="Subleading-Matched Jets PT / Subleading LJ PT"),
                   lambda objs, mask: derived_objs["subleading_matched_jets"](objs,0.4)[mask,0].pt
                       / objs["ljs"][:,1:2][mask].pt),
        ],
        evt_mask=lambda objs: (ak.num(derived_objs["subleading_matched_jets"](objs,0.4)) > 0) & (ak.num(objs["ljs"][:,1:2]) > 0),
    ),
    
    "leading_mu_matchedjet_lj_ptRatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 3.0, name="leading_mu_matchedjet_lj_ptRatio",
                   label="Leading-Mu-LJ Matched Jets PT / Leading Mu-LJ PT"),
                   lambda objs, mask: derived_objs["leading_mu_matched_jets"](objs,0.4)[mask,0].pt
                       / objs["mu_ljs"][:,0:1][mask].pt),
        ],
        evt_mask=lambda objs: (ak.num(derived_objs["leading_mu_matched_jets"](objs,0.4)) > 0) & (ak.num(objs["mu_ljs"][:,0:1]) > 0),
    ),
    
    "subleading_mu_matchedjet_lj_ptRatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 3.0, name="subleading_mu_matchedjet_lj_ptRatio",
                   label="Subleading-Mu-LJ Matched Jets PT / Subleading Mu-LJ PT"),
                   lambda objs, mask: derived_objs["subleading_mu_matched_jets"](objs,0.4)[mask,0].pt
                       / objs["mu_ljs"][:,1:2][mask].pt),
        ],
        evt_mask=lambda objs: (ak.num(derived_objs["subleading_mu_matched_jets"](objs,0.4)) > 0) & (ak.num(objs["mu_ljs"][:,1:2]) > 0),
    ),
    
    "egm_matchedjet_lj_ptRatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 3.0, name="egm_matchedjet_lj_ptRatio",
                   label="EGM-LJ Matched Jets PT / EGM-LJ PT"),
                   lambda objs, mask: derived_objs["egm_matched_jets"](objs,0.4)[mask,0].pt
                       / objs["egm_ljs"][:,0:1][mask].pt),
        ],
        evt_mask=lambda objs: (ak.num(derived_objs["egm_matched_jets"](objs,0.4)) > 0) & (ak.num(objs["egm_ljs"][:,0:1]) > 0),
    ),

    "leading_matchedjet_lj_ERatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 3.0, name="leading_matchedjet_lj_ERatio",
                   label="Leading-Matched Jets E / Leading LJ E"),
                   lambda objs, mask: derived_objs["leading_matched_jets"](objs,0.4)[mask,0].energy
                       / objs["ljs"][:,0:1][mask].energy),
        ],
        evt_mask=lambda objs: (ak.num(derived_objs["leading_matched_jets"](objs,0.4)) > 0) & (ak.num(objs["ljs"][:,0:1]) > 0),
    ),
    
    "subleading_matchedjet_lj_ERatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 3.0, name="subleading_matchedjet_lj_ERatio",
                   label="Subleading-Matched Jets E / Subleading LJ E"),
                   lambda objs, mask: derived_objs["subleading_matched_jets"](objs,0.4)[mask,0].energy
                       / objs["ljs"][:,1:2][mask].energy),
        ],
        evt_mask=lambda objs: (ak.num(derived_objs["subleading_matched_jets"](objs,0.4)) > 0) & (ak.num(objs["ljs"][:,1:2]) > 0),
    ),
    
    "leading_mu_matchedjet_lj_ERatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 3.0, name="leading_mu_matchedjet_lj_ERatio",
                   label="Leading-Mu-LJ Matched Jets E / Leading Mu-LJ E"),
                   lambda objs, mask: derived_objs["leading_mu_matched_jets"](objs,0.4)[mask,0].energy
                       / objs["mu_ljs"][:,0:1][mask].energy),
        ],
        evt_mask=lambda objs: (ak.num(derived_objs["leading_mu_matched_jets"](objs,0.4)) > 0) & (ak.num(objs["mu_ljs"][:,0:1]) > 0),
    ),
    
    "subleading_mu_matchedjet_lj_ERatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 3.0, name="subleading_mu_matchedjet_lj_ERatio",
                   label="Subleading-Mu-LJ Matched Jets E / Subleading Mu-LJ E"),
                   lambda objs, mask: derived_objs["subleading_mu_matched_jets"](objs,0.4)[mask,0].energy
                       / objs["mu_ljs"][:,1:2][mask].energy),
        ],
        evt_mask=lambda objs: (ak.num(derived_objs["subleading_mu_matched_jets"](objs,0.4)) > 0) & (ak.num(objs["mu_ljs"][:,1:2]) > 0),
    ),
    
    "egm_matchedjet_lj_ERatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 3.0, name="egm_matchedjet_lj_ERatio",
                   label="EGM-LJ Matched Jets E / EGM-LJ E"),
                   lambda objs, mask: derived_objs["egm_matched_jets"](objs,0.4)[mask,0].energy
                       / objs["egm_ljs"][:,0:1][mask].energy),
        ],
        evt_mask=lambda objs: (ak.num(derived_objs["egm_matched_jets"](objs,0.4)) > 0) & (ak.num(objs["egm_ljs"][:,0:1]) > 0),
    ),
    
    "leading_matchedjet_lep_fraction": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 1, name="leading_matchedjet_lep_fraction",
                   label="Leading-Matched Jet Lepton Fraction"),
                   lambda objs, mask: (derived_objs["leading_matched_jets"](objs,0.4)[mask].chEmEF + derived_objs["leading_matched_jets"](objs,0.4)[mask].neEmEF + derived_objs["leading_matched_jets"](objs,0.4)[mask].muEF)),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["leading_matched_jets"](objs,0.4)) > 0,
    ),
    
    "subleading_matchedjet_lep_fraction": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 1, name="subleading_matchedjet_lep_fraction",
                   label="Subleading-Matched Jet Lepton Fraction"),
                   lambda objs, mask: (derived_objs["subleading_matched_jets"](objs,0.4)[mask].chEmEF + derived_objs["subleading_matched_jets"](objs,0.4)[mask].neEmEF + derived_objs["subleading_matched_jets"](objs,0.4)[mask].muEF)),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["subleading_matched_jets"](objs,0.4)) > 0,
    ),
    
    "leading_mu_matchedjet_lep_fraction": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 1, name="leading_mu_matchedjet_lep_fraction",
                   label="Leading-Mu-LJ Matched Jet Lepton Fraction"),
                   lambda objs, mask: (derived_objs["leading_mu_matched_jets"](objs,0.4)[mask].chEmEF + derived_objs["leading_mu_matched_jets"](objs,0.4)[mask].neEmEF + derived_objs["leading_mu_matched_jets"](objs,0.4)[mask].muEF)),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["leading_mu_matched_jets"](objs,0.4)) > 0,
    ),
    
    "subleading_mu_matchedjet_lep_fraction": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 1, name="subleading_mu_matchedjet_lep_fraction",
                   label="Subleading-Mu-LJ Matched Jet Lepton Fraction"),
                   lambda objs, mask: (derived_objs["subleading_mu_matched_jets"](objs,0.4)[mask].chEmEF + derived_objs["subleading_mu_matched_jets"](objs,0.4)[mask].neEmEF + derived_objs["subleading_mu_matched_jets"](objs,0.4)[mask].muEF)),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["subleading_mu_matched_jets"](objs,0.4)) > 0,
    ),
    
    "egm_matchedjet_lep_fraction": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 1, name="egm_matchedjet_lep_fraction",
                   label="EGM-LJ Matched Jet Lepton Fraction"),
                   lambda objs, mask: (derived_objs["egm_matched_jets"](objs,0.4)[mask].chEmEF + derived_objs["egm_matched_jets"](objs,0.4)[mask].neEmEF + derived_objs["egm_matched_jets"](objs,0.4)[mask].muEF)),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["egm_matched_jets"](objs,0.4)) > 0,
    ),
    
    "leading_mu_matchedjet_chEmEF": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 1, name="leading_mu_matchedjet_chEmEF",
                   label="Leading-Mu-LJ Matched Jet chEmEF"),
                   lambda objs, mask: (derived_objs["leading_mu_matched_jets"](objs,0.4)[mask].chEmEF)),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["leading_mu_matched_jets"](objs,0.4)) > 0,
    ),
    
    "leading_mu_matchedjet_neEmEF": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 1, name="leading_mu_matchedjet_neEmEF",
                   label="Leading-Mu-LJ Matched Jet neEmEF"),
                   lambda objs, mask: (derived_objs["leading_mu_matched_jets"](objs,0.4)[mask].neEmEF)),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["leading_mu_matched_jets"](objs,0.4)) > 0,
    ),
    
    "leading_mu_matchedjet_muEF": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 1, name="leading_mu_matchedjet_muEF",
                   label="Leading-Mu-LJ Matched Jet muEF"),
                   lambda objs, mask: (derived_objs["leading_mu_matched_jets"](objs,0.4)[mask].muEF)),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["leading_mu_matched_jets"](objs,0.4)) > 0,
    ),
    
    "egm_matchedjet_chEmEF": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 1, name="egm_matchedjet_chEmEF",
                   label="EGM-LJ Matched Jet chEmEF"),
                   lambda objs, mask: (derived_objs["egm_matched_jets"](objs,0.4)[mask].chEmEF)),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["egm_matched_jets"](objs,0.4)) > 0,
    ),

    "egm_matchedjet_neEmEF": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 1, name="egm_matchedjet_neEmEF",
                   label="EGM-LJ Matched Jet neEmEF"),
                   lambda objs, mask: (derived_objs["egm_matched_jets"](objs,0.4)[mask].neEmEF)),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["egm_matched_jets"](objs,0.4)) > 0,
    ),
    
    "egm_matchedjet_muEF": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 1, name="egm_matchedjet_muEF",
                   label="EGM-LJ Matched Jet muEF"),
                   lambda objs, mask: (derived_objs["egm_matched_jets"](objs,0.4)[mask].muEF)),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["egm_matched_jets"](objs,0.4)) > 0,
    ),
    
    "leading_lj_isolation": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 2, name="leading_matchedjet_isolation",
                   label="Leading LJ Isolation"),
                   lambda objs, mask:  (derived_objs["leading_matched_jets"](objs,0.4)[mask,0].energy
                       / objs["ljs"][:,0:1][mask].energy) * (1 - (derived_objs["leading_matched_jets"](objs,0.4)[mask,0].chEmEF + 
                                                                  derived_objs["leading_matched_jets"](objs,0.4)[mask,0].neEmEF + 
                                                                  derived_objs["leading_matched_jets"](objs,0.4)[mask,0].muEF))),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["leading_matched_jets"](objs,0.4)) > 0,
    ),
    
    "new_leading_lj_isolation": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 2, name="test",
                   label="New Leading LJ Isolation"),
                   lambda objs, mask:  (derived_objs["new_leading_matched_jets"](objs,0.4)[mask][:,0:1].energy
                       / objs["ljs"][mask][:,0:1].energy) * (1 - (derived_objs["new_leading_matched_jets"](objs,0.4)[mask][:,0:1].chEmEF + 
                                                                  derived_objs["new_leading_matched_jets"](objs,0.4)[mask][:,0:1].neEmEF + 
                                                                  derived_objs["new_leading_matched_jets"](objs,0.4)[mask][:,0:1].muEF))),
        ],
        evt_mask=lambda objs: ~ak.is_none(derived_objs["new_leading_matched_jets"](objs,0.4)[:,0]),
    ),
    
    "subleading_lj_isolation": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 2, name="subleading_matchedjet_isolation",
                   label="Subleading LJ Isolation"),
                   lambda objs, mask:  (derived_objs["subleading_matched_jets"](objs,0.4)[mask,0].energy
                       / objs["ljs"][:,1:2][mask].energy) * (1 - (derived_objs["subleading_matched_jets"](objs,0.4)[mask,0].chEmEF + derived_objs["subleading_matched_jets"](objs,0.4)[mask,0].neEmEF + derived_objs["subleading_matched_jets"](objs,0.4)[mask,0].muEF))),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["subleading_matched_jets"](objs,0.4)) > 0,
    ),
    
    "new_subleading_lj_isolation": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 2, name="test",
                   label="New Subleading LJ Isolation"),
                   lambda objs, mask:  (derived_objs["new_subleading_matched_jets"](objs,0.4)[mask][:,0:1].energy
                       / objs["ljs"][mask][:,1:2].energy) * (1 - (derived_objs["new_subleading_matched_jets"](objs,0.4)[mask][:,0:1].chEmEF + 
                                                                  derived_objs["new_subleading_matched_jets"](objs,0.4)[mask][:,0:1].neEmEF + 
                                                                  derived_objs["new_subleading_matched_jets"](objs,0.4)[mask][:,0:1].muEF))),
        ],
        evt_mask=lambda objs: ~ak.is_none(derived_objs["new_subleading_matched_jets"](objs,0.4)[:,0]),
    ),
    
    "leading_mu_lj_isolation": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 2, name="leading_mu_matchedjet_isolation",
                   label="Leading Mu-LJ Isolation"),
                   lambda objs, mask:  (derived_objs["leading_mu_matched_jets"](objs,0.4)[mask,0].energy
                       / objs["mu_ljs"][:,0:1][mask].energy) * (1 - (derived_objs["leading_mu_matched_jets"](objs,0.4)[mask,0].chEmEF +
                                                                     derived_objs["leading_mu_matched_jets"](objs,0.4)[mask,0].neEmEF + 
                                                                     derived_objs["leading_mu_matched_jets"](objs,0.4)[mask,0].muEF))),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["leading_mu_matched_jets"](objs,0.4)) > 0,
    ),
    
    "new_leading_mu_lj_isolation": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 2, name="test",
                   label="New Leading Mu-LJ Isolation"),
                   lambda objs, mask:  (derived_objs["new_mu_matched_jets"](objs,0.4)[mask][:,0:1].energy
                       / objs["mu_ljs"][mask][:,0:1].energy) * (1 - (derived_objs["new_mu_matched_jets"](objs,0.4)[mask][:,0:1].chEmEF + 
                                                                  derived_objs["new_mu_matched_jets"](objs,0.4)[mask][:,0:1].neEmEF + 
                                                                  derived_objs["new_mu_matched_jets"](objs,0.4)[mask][:,0:1].muEF))),
        ],
        evt_mask=lambda objs: (ak.num(objs["mu_ljs"]) > 0) & (~ak.is_none(derived_objs["new_mu_matched_jets"](objs,0.4)[:,0])),
    ),
    
    "subleading_mu_lj_isolation": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 2, name="subleading_mu_matchedjet_isolation",
                   label="Subleading Mu-LJ Isolation"),
                   lambda objs, mask:  (derived_objs["subleading_mu_matched_jets"](objs,0.4)[mask,0].energy
                       / objs["mu_ljs"][:,1:2][mask].energy) * (1 - (derived_objs["subleading_mu_matched_jets"](objs,0.4)[mask,0].chEmEF + derived_objs["subleading_mu_matched_jets"](objs,0.4)[mask,0].neEmEF + derived_objs["subleading_mu_matched_jets"](objs,0.4)[mask,0].muEF))),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["subleading_mu_matched_jets"](objs,0.4)) > 0,
    ),
    
    "new_subleading_mu_lj_isolation": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 2, name="test",
                   label="New Subleading Mu-LJ Isolation"),
                   lambda objs, mask:  (derived_objs["new_mu_matched_jets"](objs,0.4)[ak.num(objs["mu_ljs"]) > 1][mask][:,1:2].energy
                       / objs["mu_ljs"][ak.num(objs["mu_ljs"]) > 1][mask][:,1:2].energy) * (1 - (derived_objs["new_mu_matched_jets"](objs,0.4)[ak.num(objs["mu_ljs"]) > 1][mask][:,1:2].chEmEF + 
                                                                  derived_objs["new_mu_matched_jets"](objs,0.4)[ak.num(objs["mu_ljs"]) > 1][mask][:,1:2].neEmEF + 
                                                                  derived_objs["new_mu_matched_jets"](objs,0.4)[ak.num(objs["mu_ljs"]) > 1][mask][:,1:2].muEF))),
        ],
        evt_mask=lambda objs: ~ak.is_none(derived_objs["new_mu_matched_jets"](objs,0.4)[ak.num(objs["mu_ljs"]) > 1][:,1]),
    ),
    
    "egm_lj_isolation": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 2, name="egm_matchedjet_isolation",
                   label="EGM-LJ Isolation"),
                   lambda objs, mask:  (derived_objs["egm_matched_jets"](objs,0.4)[mask,0].energy
                       / objs["egm_ljs"][:,0:1][mask].energy) * (1 - (derived_objs["egm_matched_jets"](objs,0.4)[mask,0].chEmEF + derived_objs["egm_matched_jets"](objs,0.4)[mask,0].neEmEF + derived_objs["egm_matched_jets"](objs,0.4)[mask,0].muEF))),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["egm_matched_jets"](objs,0.4)) > 0,
    ),
    
    # "new_leading_egm_lj_isolation": h.Histogram(
    #     [
    #         h.Axis(hist.axis.Regular(50, 0, 2, name="test",
    #                label="New Leading EGM-LJ Isolation"),
    #                lambda objs, mask:  (derived_objs["new_egm_matched_jets"](objs,0.4)[ak.num(objs["egm_ljs"]) > 0][mask][:,0:1].energy
    #                    / objs["egm_ljs"][ak.num(objs["egm_ljs"]) > 0][mask][:,0:1].energy) * (1 - (derived_objs["new_egm_matched_jets"](objs,0.4)[ak.num(objs["egm_ljs"]) > 0][mask][:,0:1].chEmEF + 
    #                                                               derived_objs["new_egm_matched_jets"](objs,0.4)[ak.num(objs["egm_ljs"]) > 0][mask][:,0:1].neEmEF + 
    #                                                               derived_objs["new_egm_matched_jets"](objs,0.4)[ak.num(objs["egm_ljs"]) > 0][mask][:,0:1].muEF))),
    #     ],
    #     evt_mask=lambda objs: ~ak.is_none(derived_objs["new_egm_matched_jets"](objs,0.4)[ak.num(objs["egm_ljs"]) > 0][:,0]),
    # ),
    
    "new_leading_egm_lj_isolation": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 2, name="test",
                   label="New Leading Mu-LJ Isolation"),
                   lambda objs, mask:  (derived_objs["new_egm_matched_jets"](objs,0.4)[mask][:,0:1].energy
                       / objs["egm_ljs"][mask][:,0:1].energy) * (1 - (derived_objs["new_egm_matched_jets"](objs,0.4)[mask][:,0:1].chEmEF + 
                                                                  derived_objs["new_egm_matched_jets"](objs,0.4)[mask][:,0:1].neEmEF + 
                                                                  derived_objs["new_egm_matched_jets"](objs,0.4)[mask][:,0:1].muEF))),
        ],
        evt_mask=lambda objs: (ak.num(objs["egm_ljs"]) > 0) & (~ak.is_none(derived_objs["new_egm_matched_jets"](objs,0.4)[:,0])),
    ),
    
    "leading_lj_isolation2": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 1.5, name="leading_matchedjet_isolation2",
                   label="Leading-Matched Jet Isolation2"),
                   lambda objs, mask:  abs(derived_objs["leading_matched_jets"](objs,0.4)[mask,0].energy - objs["ljs"][:,0:1][mask].energy)
                       / objs["ljs"][:,0:1][mask].energy),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["leading_matched_jets"](objs,0.4)) > 0,
    ),
    
    "subleading_lj_isolation2": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 1.5, name="subleading_matchedjet_isolation2",
                   label="Subleading-Matched Jet Isolation2"),
                   lambda objs, mask:  abs(derived_objs["subleading_matched_jets"](objs,0.4)[mask,0].energy - objs["ljs"][:,1:2][mask].energy)
                       / objs["ljs"][:,1:2][mask].energy),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["subleading_matched_jets"](objs,0.4)) > 0,
    ),
    
    "leading_mu_lj_isolation2": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 2, name="leading_mu_matchedjet_isolation2",
                   label="Leading-Mu-LJ Matched Jet Isolation2"),
                   lambda objs, mask:  abs(derived_objs["leading_mu_matched_jets"](objs,0.4)[mask,0].energy - objs["mu_ljs"][:,0:1][mask].energy)
                       / objs["mu_ljs"][:,0:1][mask].energy),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["leading_mu_matched_jets"](objs,0.4)) > 0,
    ),
    
    "subleading_mu_lj_isolation2": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 2, name="subleading_mu_matchedjet_isolation2",
                   label="Subleading-Mu-LJ Matched Jet Isolation2"),
                   lambda objs, mask:  abs(derived_objs["subleading_mu_matched_jets"](objs,0.4)[mask,0].energy - objs["mu_ljs"][:,1:2][mask].energy)
                       / objs["mu_ljs"][:,1:2][mask].energy),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["subleading_mu_matched_jets"](objs,0.4)) > 0,
    ),
    
    "egm_lj_isolation2": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 2, name="egm_matchedjet_isolation2",
                   label="EGM-LJ Matched Jet Isolation2"),
                   lambda objs, mask:  abs(derived_objs["egm_matched_jets"](objs,0.4)[mask,0].energy - objs["egm_ljs"][:,0:1][mask].energy)
                       / objs["egm_ljs"][:,0:1][mask].energy),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["egm_matched_jets"](objs,0.4)) > 0,
    ),
    
    "jet_lep_fraction": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 1, name="jet_lep_fraction",
                   label="Jet Lepton Fraction"),
                   lambda objs, mask: (objs["jets"].chEmEF + objs["jets"].neEmEF + 
                                       objs["jets"].muEF)),
        ],
    ),
    
    "jet1_lep_fraction": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 1, name="jet1_lep_fraction",
                   label="Leading Jet Lepton Fraction"),
                   lambda objs, mask: (objs["jets"][mask,0].chEmEF + objs["jets"][mask,0].neEmEF + 
                                       objs["jets"][mask,0].muEF)),
        ],
        evt_mask=lambda objs: ak.num(objs["jets"]) > 0,
    ),
    
    "jet2_lep_fraction": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 1, name="jet2_lep_fraction",
                   label="Subleading Jet Lepton Fraction"),
                   lambda objs, mask: (objs["jets"][mask,1].chEmEF + objs["jets"][mask,1].neEmEF + 
                                       objs["jets"][mask,1].muEF)),
        ],
        evt_mask=lambda objs: ak.num(objs["jets"]) > 1,
    ),
    
    "jet3_lep_fraction": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 1, name="jet3_lep_fraction",
                   label="Third Jet Lepton Fraction"),
                   lambda objs, mask: (objs["jets"][mask,2].chEmEF + objs["jets"][mask,2].neEmEF + 
                                       objs["jets"][mask,2].muEF)),
        ],
        evt_mask=lambda objs: ak.num(objs["jets"]) > 2,
    ),
    
    "jet4_lep_fraction": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 1, name="jet4_lep_fraction",
                   label="Fourth Jet Lepton Fraction"),
                   lambda objs, mask: (objs["jets"][mask,3].chEmEF + objs["jets"][mask,3].neEmEF + 
                                       objs["jets"][mask,3].muEF)),
        ],
        evt_mask=lambda objs: ak.num(objs["jets"]) > 3,
    ),
    
    "jet5_lep_fraction": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 1, name="jet5_lep_fraction",
                   label="Fifth Jet Lepton Fraction"),
                   lambda objs, mask: (objs["jets"][mask,4].chEmEF + objs["jets"][mask,4].neEmEF + 
                                       objs["jets"][mask,4].muEF)),
        ],
        evt_mask=lambda objs: ak.num(objs["jets"]) > 4,
    ),
    
    "jet_nmuon": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="Number of Muon in Jet"),
                   lambda objs, mask: objs["jets"].nMuons),
        ],
    ),
    
    "jet1_nmuon": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="Number of Muon in Leading Jet"),
                   lambda objs, mask: objs["jets"][mask,0].nMuons),
        ],
        evt_mask=lambda objs: ak.num(objs["jets"]) > 0,
    ),
    
    "jet2_nmuon": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="Number of Muon in Subleading Jet"),
                   lambda objs, mask: objs["jets"][mask,1].nMuons),
        ],
        evt_mask=lambda objs: ak.num(objs["jets"]) > 1,
    ),
    
    "jet3_nmuon": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="Number of Muon in Third Jet"),
                   lambda objs, mask: objs["jets"][mask,2].nMuons),
        ],
        evt_mask=lambda objs: ak.num(objs["jets"]) > 2,
    ),
    
    "jet4_nmuon": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="Number of Muon in Fourth Jet"),
                   lambda objs, mask: objs["jets"][mask,3].nMuons),
        ],
        evt_mask=lambda objs: ak.num(objs["jets"]) > 3,
    ),
    
    "jet5_nmuon": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="Number of Muon in Fifth Jet"),
                   lambda objs, mask: objs["jets"][mask,4].nMuons),
        ],
        evt_mask=lambda objs: ak.num(objs["jets"]) > 4,
    ),
    
    "jet_nelectron": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="Number of Electron in Jet"),
                   lambda objs, mask: objs["jets"].nElectrons),
        ],
    ),
    
    "jet1_nelectron": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="Number of Electron in Leading Jet"),
                   lambda objs, mask: objs["jets"][mask,0].nElectrons),
        ],
        evt_mask=lambda objs: ak.num(objs["jets"]) > 0,
    ),
    
    "jet2_nelectron": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="Number of Electron in Subleading Jet"),
                   lambda objs, mask: objs["jets"][mask,1].nElectrons),
        ],
        evt_mask=lambda objs: ak.num(objs["jets"]) > 1,
    ),
    
    "jet3_nelectron": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="Number of Electron in Third Jet"),
                   lambda objs, mask: objs["jets"][mask,2].nElectrons),
        ],
        evt_mask=lambda objs: ak.num(objs["jets"]) > 2,
    ),
    
    "jet4_nelectron": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="Number of Electron in Fourth Jet"),
                   lambda objs, mask: objs["jets"][mask,3].nElectrons),
        ],
        evt_mask=lambda objs: ak.num(objs["jets"]) > 3,
    ),
    
    "jet5_nelectron": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="Number of Electron in Fifth Jet"),
                   lambda objs, mask: objs["jets"][mask,4].nElectrons),
        ],
        evt_mask=lambda objs: ak.num(objs["jets"]) > 4,
    ),
    
    "leading_mu_matchedjet_nmuon": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="Number of Muon in Leading Mu-LJ Matched Jet"),
                   lambda objs, mask: derived_objs["leading_mu_matched_jets"](objs,0.4)[mask].nMuons),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["leading_mu_matched_jets"](objs,0.4)) > 0,
    ),
    
    "subleading_mu_matchedjet_nmuon": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="Number of Muon in Subleading Mu-LJ Matched Jet"),
                   lambda objs, mask: derived_objs["subleading_mu_matched_jets"](objs,0.4)[mask].nMuons),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["subleading_mu_matched_jets"](objs,0.4)) > 0,
    ),
    
    "egm_matchedjet_nmuon": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="Number of Muon in EGM-LJ Matched Jet"),
                   lambda objs, mask: derived_objs["egm_matched_jets"](objs,0.4)[mask].nMuons),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["egm_matched_jets"](objs,0.4)) > 0,
    ),
    
    "leading_mu_matchedjet_nelectron": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="Number of Electron in Leading Mu-LJ Matched Jet"),
                   lambda objs, mask: derived_objs["leading_mu_matched_jets"](objs,0.4)[mask].nElectrons),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["leading_mu_matched_jets"](objs,0.4)) > 0,
    ),
    
    "subleading_mu_matchedjet_nelectron": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="Number of Electron in Subleading Mu-LJ Matched Jet"),
                   lambda objs, mask: derived_objs["subleading_mu_matched_jets"](objs,0.4)[mask].nElectrons),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["subleading_mu_matched_jets"](objs,0.4)) > 0,
    ),
    
    "egm_matchedjet_nelectron": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="Number of Electron in EGM-LJ Matched Jet"),
                   lambda objs, mask: derived_objs["egm_matched_jets"](objs,0.4)[mask].nElectrons),
        ],
        evt_mask=lambda objs: ak.num(derived_objs["egm_matched_jets"](objs,0.4)) > 0,
    ),
    
    # pv
    "pv_n": obj_attr("pvs", "npvs", nbins=50, label="Number of PVs"),
    "pv_ndof": obj_attr("pvs", "ndof", nbins=25, xmax=100),
    "pv_z": obj_attr("pvs", "z", xmin=-50, xmax=50),
    "pv_rho": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, -0.5, 0.5, name="pv_rho"),
                   lambda objs, mask: objs["pvs"].pos.rho),
        ],
    ),
    # GSFelectron: Plottting electron ID varaiables and plotting 2D hists of the leading electron
    # ID variables in barrel within Delta R < .5 of a dark photon vs the lxy of the dark photon
    "electron_GsfEleDEtaInSeedCut": h.Histogram(
        [
            h.Axis(hist.axis.Regular(35, 0, .0070, name="electron_GsfEleDEtaInSeedCut"),
                   lambda objs, mask: objs["electrons"].GsfEleDEtaInSeedCut_0),
        ],
    ),
    "electron_GsfEleDEtaInSeedCut2d": h.Histogram(
        [  # lxy of dark photon that decays to electrons
            h.Axis(hist.axis.Regular(100, 0, 500, name="genA_lxy",
                                     label=r"Dark photon $L_{xy}$ [cm]"),
                   lambda objs, mask: lxy(objs["genAs_toE"])[mask]),
            h.Axis(hist.axis.Regular(35, 0, .0070, name="electron_GsfEleDEtaInSeedCut"),
                   lambda objs, mask: matched(objs["electrons"], objs["genAs_toE"], 0.5)[mask, 0:1].GsfEleDEtaInSeedCut_0)
        ],
        evt_mask = lambda objs: ak.num(matched(objs["electrons"], objs["genAs_toE"], 0.5)) > 0,
    ),
    "electron_GsfEleDPhiInCut": h.Histogram(
        [
            h.Axis(hist.axis.Regular(45, 0, .0450, name="electron_GsfEleDPhiInCut"),
                   lambda objs, mask: objs["electrons"].GsfEleDPhiInCut_0),
        ],
    ),
    "electron_GsfEleDPhiInCut2d": h.Histogram(
        [  # lxy of dark photon that decays to electrons
            h.Axis(hist.axis.Regular(100, 0, 500, name="genA_lxy",
                                     label=r"Dark photon $L_{xy}$ [cm]"),
                   lambda objs, mask: lxy(objs["genAs_toE"])[mask]),
            h.Axis(hist.axis.Regular(45, 0, .09, name="electron_GsfEleDPhiInCut"),
                   lambda objs, mask: matched(objs["electrons"], objs["genAs_toE"], 0.5)[mask, 0:1].GsfEleDPhiInCut_0)
        ],
        evt_mask = lambda objs: ak.num(matched(objs["electrons"], objs["genAs_toE"], 0.5)) > 0,
    ),
    "electron_GsfEleEInverseMinusPInverseCut": h.Histogram(
        [
            h.Axis(hist.axis.Regular(60, 0, .3, name="electron_GsfEleEInverseMinusPInverseCut"),
                   lambda objs, mask: objs["electrons"].GsfEleEInverseMinusPInverseCut_0),
        ],
    ),
    "electron_GsfEleEInverseMinusPInverseCut2d": h.Histogram(
        [  # lxy of dark photon that decays to electrons
            h.Axis(hist.axis.Regular(100, 0, 500, name="genA_lxy",
                                     label=r"Dark photon $L_{xy}$ [cm]"),
                   lambda objs, mask: lxy(objs["genAs_toE"])[mask]),
            h.Axis(hist.axis.Regular(60, 0, .3, name="electron_GsfEleEInverseMinusPInverseCut"),
                   lambda objs, mask: matched(objs["electrons"], objs["genAs_toE"], 0.5)[mask, 0:1].GsfEleEInverseMinusPInverseCut_0)
        ],
        evt_mask = lambda objs: ak.num(matched(objs["electrons"], objs["genAs_toE"], 0.5)) > 0,
    ),
    "electron_GsfEleRelPFIsoScaledCut": h.Histogram(
        [
            h.Axis(hist.axis.Regular(40, 0, .2, name="electron_GsfEleRelPFIsoScaledCut"),
                   lambda objs, mask: (objs["electrons"].GsfEleRelPFIsoScaledCut_0
                                       - .506/objs["electrons"].pt)),
        ],
    ),
    "electron_GsfEleRelPFIsoScaledCut2d": h.Histogram(
        [  # lxy of dark photon that decays to electrons
           # added the alegbra relIso has in the analysis note
            h.Axis(hist.axis.Regular(100, 0, 500, name="genA_lxy",
                                     label=r"Dark photon $L_{xy}$ [cm]"),
                   lambda objs, mask: lxy(objs["genAs_toE"])[mask]),
            h.Axis(hist.axis.Regular(40, 0, .2, name="electron_GsfEleRelPFIsoScaledCut"),
                   lambda objs, mask: (matched(objs["electrons"], objs["genAs_toE"], 0.5)[mask, 0:1].GsfEleRelPFIsoScaledCut_0
                                       - .506/(matched(objs["electrons"], objs["genAs_toE"], 0.5)[mask, 0:1].pt))),
        ],
        evt_mask = lambda objs: ak.num(matched(objs["electrons"], objs["genAs_toE"], 0.5)) > 0,
    ),
    "electron_GsfEleFull5x5SigmaIEtaIEtaCut": h.Histogram(
        [
            h.Axis(hist.axis.Regular(45, 0, .045, name="electron_GsfEleFull5x5SigmaIEtaIEtaCut"),
                   lambda objs, mask: objs["electrons"].GsfEleFull5x5SigmaIEtaIEtaCut_0),
        ],
    ),
    "electron_GsfEleFull5x5SigmaIEtaIEtaCut2d": h.Histogram(
        [  # lxy of dark photon that decays to electrons
            h.Axis(hist.axis.Regular(100, 0, 500, name="genA_lxy",
                                     label=r"Dark photon $L_{xy}$ [cm]"),
                   lambda objs, mask: lxy(objs["genAs_toE"])[mask]),
            h.Axis(hist.axis.Regular(45, 0, .045, name="electron_GsfEleFull5x5SigmaIEtaIEtaCut"),
                   lambda objs, mask: matched(objs["electrons"], objs["genAs_toE"], 0.5)[mask, 0:1].GsfEleFull5x5SigmaIEtaIEtaCut_0)
        ],
        evt_mask = lambda objs: ak.num(matched(objs["electrons"], objs["genAs_toE"], 0.5)) > 0,
    ),
    "electron_GsfEleConversionVetoCut": h.Histogram(
        [
            h.Axis(hist.axis.Regular(2, 0, 2, name="electron_GsfEleConversionVetoCut"),
                   lambda objs, mask: objs["electrons"].GsfEleConversionVetoCut_0),
        ],
    ),
    "electron_GsfEleConversionVetoCut2d": h.Histogram(
        [  # lxy of dark photon that decays to electrons
            h.Axis(hist.axis.Regular(100, 0, 500, name="genA_lxy",
                                     label=r"Dark photon $L_{xy}$ [cm]"),
                   lambda objs, mask: lxy(objs["genAs_toE"])[mask]),
            h.Axis(hist.axis.Regular(2, 0, 2, name="electron_GsfEleConversionVetoCut"),
                   lambda objs, mask: matched(objs["electrons"], objs["genAs_toE"], 0.5)[mask, 0:1].GsfEleConversionVetoCut_0)
        ],
        evt_mask = lambda objs: ak.num(matched(objs["electrons"], objs["genAs_toE"], 0.5)) > 0,
    ),
    "electron_GsfEleHadronicOverEMEnergyScaledCut": h.Histogram(
         [
             h.Axis(hist.axis.Regular(30, 0, .15, name="electron_GsfEleHadronicOverEMEnergyScaledCut"),
                    lambda objs, mask: objs["electrons"].GsfEleHadronicOverEMEnergyScaledCut_0),
         ],
     ),
    "electron_GsfEleHadronicOverEMEnergyScaledCut2d": h.Histogram(
        [  # lxy of dark photon that decays to electrons
            h.Axis(hist.axis.Regular(100, 0, 500, name="genA_lxy",
                                     label=r"Dark photon $L_{xy}$ [cm]"),
                   lambda objs, mask: lxy(objs["genAs_toE"])[mask]),
            h.Axis(hist.axis.Regular(30, 0, .15, name="electron_GsfEleHadronicOverEMEnergyScaledCut"),
                   lambda objs, mask: matched(objs["electrons"], objs["genAs_toE"], 0.5)[mask, 0:1].GsfEleHadronicOverEMEnergyScaledCut_0)
        ],
        evt_mask = lambda objs: ak.num(matched(objs["electrons"], objs["genAs_toE"], 0.5)) > 0,
    ),
    "electron_GsfEleMissingHitsCut": h.Histogram(
        [
            h.Axis(hist.axis.Regular(10, 0, 10, name="electron_GsfEleMissingHitsCut"),
                   lambda objs, mask: objs["electrons"].GsfEleMissingHitsCut_0),
        ],
    ),
    "electron_GsfEleMissingHitsCut2d": h.Histogram(
        [  # lxy of dark photon that decays to electrons
            h.Axis(hist.axis.Regular(100, 0, 500, name="genA_lxy",
                                     label=r"Dark photon $L_{xy}$ [cm]"),
                   lambda objs, mask: lxy(objs["genAs_toE"])[mask]),
            h.Axis(hist.axis.Regular(10, 0, 10, name="electron_GsfEleMissingHitsCut"),
                   lambda objs, mask: matched(objs["electrons"], objs["genAs_toE"], 0.5)[mask, 0:1].GsfEleMissingHitsCut_0)
        ],
        evt_mask = lambda objs: ak.num(matched(objs["electrons"], objs["genAs_toE"], 0.5)) > 0,
    ),
    # pfelectron
    "electron_n": obj_attr("electrons", "n"),
    "electron_pt": obj_attr("electrons", "pt"),
    "electron_eta_phi": obj_eta_phi("electrons"),
    "electron_nearGenA_n": h.Histogram(
        [
            # number of electrons within dR=0.5 of a genA that decays to electrons
            h.Axis(hist.axis.Integer(0, 10, name="electron_nearGenA_n"),
                   lambda objs, mask: ak.num(matched(objs["electrons"], objs["genAs_toE"], 0.5))),
        ],
    ),
    # pfelectron-genA
    "electron_nearGenA_n_genA_lxy": h.Histogram(
        [
            # lxy of dark photon that decays to electrons
            h.Axis(hist.axis.Regular(100, 0, 500, name="genA_lxy",
                                     label=r"Dark photon $L_{xy}$ [cm]"),
                   lambda objs, mask: lxy(objs["genAs_toE"])),
            # number of electrons within dR=0.5 of a genA that decays to electrons
            h.Axis(hist.axis.Integer(0, 4, name="electron_nearGenA_n", label="$N_{e}$"),
                   lambda objs, mask: ak.num(matched(objs["electrons"], objs["genAs_toE"], 0.5))),
        ],
    ),
    # pfelectron-genElectron
    "electron_genE_dR": h.Histogram(
        [
            # dR(e, nearest gen e)
            h.Axis(hist.axis.Regular(50, 0, 2*math.pi, name="electron_genE_dR"),
                   lambda objs, mask: dR(objs["electrons"], objs["genEs"]))
        ],
    ),
    # pfphoton
    "photon_n": obj_attr("photons", "n"),
    "photon_pt":obj_attr("photons", "pt"),
    "photon_eta_phi": obj_eta_phi("photons"),
    "photon_nearGenA_n": h.Histogram(
        [
            # number of photons within dR=0.5 of a genA that decays to electrons
            h.Axis(hist.axis.Integer(0, 10, name="photon_nearGenA_n"),
                   lambda objs, mask: ak.num(matched(objs["photons"], objs["genAs_toE"], 0.5))),
        ],
    ),
    # pfphoton-genA
    "photon_nearGenA_n_genA_lxy": h.Histogram(
        [
            # lxy of dark photon that decays to electrons
            h.Axis(hist.axis.Regular(100, 0, 500, name="genA_lxy",
                                     label=r"Dark photon $L_{xy}$ [cm]"),
                   lambda objs, mask: lxy(objs["genAs_toE"])),
            # number of photons within dR=0.5 of a genA that decays to electrons
            h.Axis(hist.axis.Integer(0, 4, name="photon_nearGenA_n", label=r"$N_{\gamma}$"),
                   lambda objs, mask: ak.num(matched(objs["photons"], objs["genAs_toE"], 0.5))),
        ],
    ),
    # pfphoton-genElectron
    "photon_genE_dR": h.Histogram(
        [
            # dR(photon, nearest gen e)
            h.Axis(hist.axis.Regular(50, 0, 2*math.pi, name="photon_genE_dR"),
                   lambda objs, mask: dR(objs["photons"], objs["genEs"]))
        ],
    ),
    # pfmuon
    "muon_n": obj_attr("muons", "n"),
    "muon_pt":obj_attr("muons", "pt"),
    "muon_eta_phi": obj_eta_phi("muons"),
    "muon_absD0": obj_attr("muons", "dxy", absval=True, xmax=500),
    "muon_absD0_lowRange": obj_attr("muons", "dxy", absval=True, xmax=10),
    "muon_nearGenA_n": h.Histogram(
        [
            # number of muons within dR=0.5 of a genA that decays to muons
            h.Axis(hist.axis.Integer(0, 10, name="muon_nearGenA_n"),
                   lambda objs, mask: ak.num(matched(objs["muons"], objs["genAs_toMu"], 0.5))),
        ],
    ),
    # pfmuon-genA
    "muon_nearGenA_n_genA_lxy": h.Histogram(
        [
            # lxy of dark photon that decays to muons
            h.Axis(hist.axis.Regular(100, 0, 500, name="genA_lxy",
                                     label=r"Dark photon $L_{xy}$ [cm]"),
                   lambda objs, mask: lxy(objs["genAs_toMu"])),
            # number of muons within dR=0.5 of a genA that decays to muons
            h.Axis(hist.axis.Integer(0, 4, name="muon_nearGenA_n", label=r"$N_{\mu^{PF}}$"),
                   lambda objs, mask: ak.num(matched(objs["muons"], objs["genAs_toMu"], 0.5))),
        ],
    ),
    # pfmuon-genMuon
    "muon_genMu_dR": h.Histogram(
        [
            # dR(mu, nearest gen mu)
            h.Axis(hist.axis.Regular(50, 0, 2*math.pi, name="muon_genMu_dR"),
                   lambda objs, mask: dR(objs["muons"], objs["genMus"]))
        ],
    ),
    # dsamuon
    "dsaMuon_n": obj_attr("dsaMuons", "n"),
    "dsaMuon_pt":obj_attr("dsaMuons", "pt"),
    "dsaMuon_eta_phi": obj_eta_phi("dsaMuons"),
    "dsaMuon_absD0": obj_attr("dsaMuons", "dxy", absval=True, xmax=500),
    "dsaMuon_absD0_lowRange": obj_attr("dsaMuons", "dxy", absval=True, xmax=10),
    "dsaMuon_nearGenA_n": h.Histogram(
        [
            # number of muons within dR=0.5 of a genA that decays to muons
            h.Axis(hist.axis.Integer(0, 10, name="dsaMuon_nearGenA_n"),
                   lambda objs, mask: ak.num(matched(objs["dsaMuons"], objs["genAs_toMu"], 0.5))),
        ],
    ),
    # dsamuon-genA
    "dsaMuon_nearGenA_n_genA_lxy": h.Histogram(
        [
            # lxy of dark photon that decays to dsaMuons
            h.Axis(hist.axis.Regular(100, 0, 500, name="genA_lxy",
                                     label=r"Dark photon $L_{xy}$ [cm]"),
                   lambda objs, mask: lxy(objs["genAs_toMu"])),
            # number of dsaMuons within dR=0.5 of a genA that decays to muons
            h.Axis(hist.axis.Integer(0, 4, name="dsaMuon_nearGenA_n", label=r"$N_{\mu^{DSA}}$"),
                   lambda objs, mask: ak.num(matched(objs["dsaMuons"], objs["genAs_toMu"], 0.5))),
        ],
    ),
    # dsaMuon-genMuon
    "dsaMuon_genMu_dR": h.Histogram(
        [
            # dR(dsa mu, nearest gen mu)
            h.Axis(hist.axis.Regular(50, 0, 2*math.pi, name="dsaMuon_genMu_dR"),
                   lambda objs, mask: dR(objs["dsaMuons"], objs["genMus"]))
        ],
    ),
    # lj
    "lj_n": obj_attr("ljs", "n"),
    "egm_lj_n": obj_attr("egm_ljs", "n"),
    "electron_lj_n": obj_attr("electron_ljs", "n"),    
    "photon_lj_n": obj_attr("photon_ljs", "n"),
    "leading_egm_lj_n": h.Histogram(
        [
            h.Axis(hist.axis.Regular(10, 0, 10, name="leading_egm_lj_n", label="Number of EGM-LJ"),
                   lambda objs, mask: ak.num(objs["egm_ljs"][:,0:1])),
        ],
    ),
    "mu_lj_n": obj_attr("mu_ljs", "n"),
    "pfmu_lj_n": obj_attr("pfmu_ljs", "n"),
    "dsamu_lj_n": obj_attr("dsamu_ljs", "n"),
    "leading_mu_lj_n": h.Histogram(
        [
            h.Axis(hist.axis.Regular(10, 0, 10, name="leading_mu_lj_n", label="Number of Leading Mu-LJ"),
                   lambda objs, mask: ak.num(objs["mu_ljs"][:,0:1])),
        ],
    ),
    "subleading_mu_lj_n": h.Histogram(
        [
            h.Axis(hist.axis.Regular(10, 0, 10, name="subleading_mu_lj_n", label="Number of Subleading Mu-LJ"),
                   lambda objs, mask: ak.num(objs["mu_ljs"][:,1:2])),
        ],
    ),   
    "lj_pt": obj_attr("ljs", "pt", xmax=500),
    "lj0_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 500, name="lj0_pt",
                                     label="Leading lepton jet pT [GeV]"),
                   lambda objs, mask: objs["ljs"][mask, 0].pt),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 0,
    ),
    "lj1_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 500, name="lj1_pt",
                                     label="Subleading lepton jet pT [GeV]"),
                   lambda objs, mask: objs["ljs"][mask, 1].pt),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 1,
    ),
    
    "lj2_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 500, name="lj2_pt",
                                     label="Third lepton jet pT [GeV]"),
                   lambda objs, mask: objs["ljs"][mask, 2].pt),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 2,
    ),

    "lj3_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 500, name="lj3_pt",
                                     label="Fourth lepton jet pT [GeV]"),
                   lambda objs, mask: objs["ljs"][mask, 3].pt),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 3,
    ),
    
    "lj0_e": h.Histogram(
        [
            h.Axis(hist.axis.Regular(350, 0, 700, name="lj_e",
                                     label="Leading lepton jet E [GeV]"),
                   lambda objs, mask: objs["ljs"][mask, 0].energy),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 0,
    ),
    "lj1_e": h.Histogram(
        [
            h.Axis(hist.axis.Regular(350, 0, 700, name="lj_e",
                                     label="Subleading lepton jet E [GeV]"),
                   lambda objs, mask: objs["ljs"][mask, 1].energy),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 1,
    ),
    "lj0_dRSpread": h.Histogram(
        [
            h.Axis(hist.axis.Regular(250, 0, 1.0, name="lj0_dRSpread",
                                     label="Leading lepton jet dRSpread"),
                   lambda objs, mask: objs["ljs"][mask, 0].dRSpread),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 0,
    ),
    "lj1_dRSpread": h.Histogram(
        [
            h.Axis(hist.axis.Regular(250, 0, 1.0, name="lj1_dRSpread",
                                     label="Subleading lepton jet dRSpread"),
                   lambda objs, mask: objs["ljs"][mask, 1].dRSpread),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 1,
    ),
    "lj_eta_phi": obj_eta_phi("ljs"),
    "lj_eta": obj_attr("ljs", "eta", xmax=3),
    "lj_phi": obj_attr("ljs", "phi", xmax=math.pi),
    "egm_lj_pt": obj_attr("egm_ljs", "pt", xmax=500),
    "mu_lj_pt": obj_attr("mu_ljs", "pt", xmax=500),   
    "leading_mu_lj_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 500, name="leading_mu_lj_pt", label="Leading Mu-LJ PT [GeV]"),
                   lambda objs, mask: objs["mu_ljs"][:,0:1][mask].pt),
        ],
        evt_mask=lambda objs: ak.num(objs["mu_ljs"]) > 0,
    ),
    "subleading_mu_lj_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 500, name="subleading_mu_lj_pt", label="Subleading Mu-LJ PT [GeV]"),
                   lambda objs, mask: objs["mu_ljs"][:,1:2][mask].pt),
        ],
        evt_mask=lambda objs: ak.num(objs["mu_ljs"]) > 1,
    ),
    "leading_egm_lj_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 500, name="leading_egm_lj_pt", label="EGM-LJ PT [GeV]"),
                   lambda objs, mask: objs["egm_ljs"][:,0:1][mask].pt),
        ],
        evt_mask=lambda objs: ak.num(objs["egm_ljs"]) > 0,
    ),  
    "lj_electronN": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="lj_electronN", label="Number of Electrons in LJ"),
                   lambda objs, mask: objs["ljs"].electron_n),
        ],
    ),
    "mu_lj_muonN": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="mu_lj_muonN"),
                   lambda objs, mask: objs["mu_ljs"].muon_n),
        ],
    ),
    "egm_lj_electronN": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="egm_lj_electronN"),
                   lambda objs, mask: objs["egm_ljs"].electron_n),
        ],
    ),
    "egm_lj_photonN": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="egm_lj_photonN"),
                   lambda objs, mask: objs["egm_ljs"].photon_n),
        ],
    ),
    "lj_photonN": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="lj_photonN", label="Number of Photons in LJ"),
                   lambda objs, mask: objs["ljs"].photon_n),
        ],
    ),
    "lj_electronPhotonN": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="lj_electronPhotonN", label="Number of (Electrons + Photons) in LJ"),
                   lambda objs, mask: objs["ljs"].electron_n + objs["ljs"].photon_n),
        ],
    ),
    "lj_muonN": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="lj_muonN"),
                   lambda objs, mask: objs["ljs"].muon_n),
        ],
    ),
    "lj_dsaMuN": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="lj_dsaMuN", label="Number of DSA Muons in LJ"),
                   lambda objs, mask: objs["ljs"].dsaMu_n),
        ],
    ),
    "lj_pfMuN": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 10, name="lj_pfMuN", label="Number of PF Muons in LJ"),
                   lambda objs, mask: objs["ljs"].pfMu_n),
        ],
    ),
    # pfelectron-lj
    "electron_lj_dR": h.Histogram(
        [
            # dR(e, nearest LJ)
            h.Axis(hist.axis.Regular(100, 0, 2*math.pi, name="electron_lj_dR"),
                   lambda objs, mask: dR(objs["electrons"], objs["ljs"]))
        ],
    ),
    "electron_lj_dR_lowRange": h.Histogram(
        [
            # dR(e, nearest LJ)
            h.Axis(hist.axis.Regular(100, 0, 1.0, name="electron_lj_dR_lowRange"),
                   lambda objs, mask: dR(objs["electrons"], objs["ljs"]))
        ],
    ),
    # pfphoton-lj
    "photon_lj_dR": h.Histogram(
        [
            # dR(e, nearest LJ)
            h.Axis(hist.axis.Regular(100, 0, 2*math.pi, name="photon_lj_dR"),
                   lambda objs, mask: dR(objs["photons"], objs["ljs"]))
        ],
    ),
    "photon_lj_dR_lowRange": h.Histogram(
        [
            # dR(photon, nearest LJ)
            h.Axis(hist.axis.Regular(100, 0, 1.0, name="photon_lj_dR_lowRange"),
                   lambda objs, mask: dR(objs["photons"], objs["ljs"]))
        ],
    ),
    "photon_lj_dR_reallyLowRange": h.Histogram(
        [
            # dR(photon, nearest LJ)
            h.Axis(hist.axis.Regular(100, 0, 0.1, name="photon_lj_dR_reallyLowRange"),
                   lambda objs, mask: dR(objs["photons"], objs["ljs"]))
        ],
    ),
    # pfmuon-lj
    "muon_lj_dR": h.Histogram(
        [
            # dR(mu, nearest LJ)
            h.Axis(hist.axis.Regular(100, 0, 2*math.pi, name="muon_lj_dR"),
                   lambda objs, mask: dR(objs["muons"], objs["ljs"]))
        ],
    ),
    "muon_lj_dR_lowRange": h.Histogram(
        [
            # dR(mu, nearest LJ)
            h.Axis(hist.axis.Regular(100, 0, 1.0, name="muon_lj_dR_lowRange"),
                   lambda objs, mask: dR(objs["muons"], objs["ljs"]))
        ],
    ),
    # dsamuon-lj
    "dsaMuon_lj_dR": h.Histogram(
        [
            # dR(dsa mu, nearest LJ)
            h.Axis(hist.axis.Regular(100, 0, 2*math.pi, name="dsaMuon_lj_dR"),
                   lambda objs, mask: dR(objs["dsaMuons"], objs["ljs"]))
        ],
    ),
    "dsaMuon_lj_dR_lowRange": h.Histogram(
        [
            # dR(dsa mu, nearest LJ)
            h.Axis(hist.axis.Regular(50, 0, 1.0, name="dsaMuon_lj_dR_lowRange"),
                   lambda objs, mask: dR(objs["dsaMuons"], objs["ljs"]))
        ],
    ),
    # lj-lj
    "mu_lj_egm_lj_invmass": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 1200, name="mu_lj_egm_lj_invmass",
                                     label=r"Invariant Mass (Mu-LJ, EGM-LJ)"),
                   lambda objs, mask: (objs["mu_ljs"][mask,0] + objs["egm_ljs"][mask,0]).mass),
        ],
        evt_mask=lambda objs: (ak.num(objs["mu_ljs"]) > 0) & (ak.num(objs["egm_ljs"]) > 0),
    ),
    
    "mu_lj_mu_lj_invmass": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 1200, name="mu_lj_mu_lj_invmass",
                                     label=r"Invariant Mass (Mu-LJ, Mu-LJ)"),
                   lambda objs, mask: (objs["mu_ljs"][mask,0] + objs["mu_ljs"][mask,1]).mass),
        ],
        evt_mask=lambda objs: (ak.num(objs["mu_ljs"]) > 1),
    ),
    "lj_lj_absdphi": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 2*math.pi, name="|$\Delta\phi$| (Leading LJ, Subleading LJ)"),
                   lambda objs, mask: abs(objs["ljs"][mask, 1].phi - objs["ljs"][mask, 0].phi)),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 1,
    ),
    "lj_lj_absdR": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 6, name="|$\Delta$R| ($LJ_{0}$, $LJ_{1}$)"),
                   lambda objs, mask: objs["ljs"][mask, 1].delta_r(objs["ljs"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 1,
    ),
    "lj_lj_absdeta": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 6, name="|$\Delta\eta$| ($LJ_{0}$, $LJ_{1}$)"),
                   lambda objs, mask: abs(objs["ljs"][mask, 1].eta - objs["ljs"][mask, 0].eta)),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 1,
    ),
    "lj_lj_invmass": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 1200, name="ljlj_mass",
                                     label=r"Invariant Mass (Leading LJ, Subleading LJ)"),
                   lambda objs, mask: objs["ljs"][mask, :2].sum().mass),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 1,
    ),
    "lj_lj_invmass_lowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 500, name="ljlj_mass",
                                     label=r"Invariant Mass (Leading LJ, Subleading LJ)"),
                   lambda objs, mask: objs["ljs"][mask, :2].sum().mass),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 1,
    ),
    "lj_lj_ptRatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 1.0, 2.0, name="lj_lj_ptRatio",
                   label="Leading LJ PT / Subleading LJ PT"),
                   lambda objs, mask: objs["ljs"][mask, 0].pt / objs["ljs"][mask, 1].pt),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 1,
    ),
    # ABCD plane
    "lj_lj_absdphi_iso1": h.Histogram(
        [
            h.Axis(hist.axis.Regular(20, 0, math.pi, name="|$\Delta\phi$| ($LJ_{0}$, $LJ_{1}$)"),
                   lambda objs, mask: abs(objs["ljs"][mask, 1].phi - objs["ljs"][mask, 0].phi)),
            h.Axis(hist.axis.Regular(20, 0, 1, name="leading_mu_matchedjet_isolation",
                                     label=r"Leading-Mu-LJ Matched Jet Isolation"),
                   lambda objs, mask: (derived_objs["leading_mu_matched_jets"](objs,0.4)[mask,0].energy
                       / objs["mu_ljs"][:,0:1][mask].energy) * (1 - (derived_objs["leading_mu_matched_jets"](objs,0.4)[mask,0].chEmEF + 
                                                                                    derived_objs["leading_mu_matched_jets"](objs,0.4)[mask,0].neEmEF + 
                                                                                    derived_objs["leading_mu_matched_jets"](objs,0.4)[mask,0].muEF))),
        ],
        evt_mask=lambda objs: ak.num(abs(objs["ljs"][:,0:1].phi - objs["ljs"][:,1:2].phi)) == ak.num(derived_objs["leading_mu_matched_jets"](objs,0.4)),
    ),

    "lj_lj_absdphi_iso2": h.Histogram(
        [
            h.Axis(hist.axis.Regular(20, 0, math.pi, name="|$\Delta\phi$| ($LJ_{0}$, $LJ_{1}$)"),
                   lambda objs, mask: abs(objs["ljs"][mask, 1].phi - objs["ljs"][mask, 0].phi)),
            h.Axis(hist.axis.Regular(20, 0, 1, name="leading_mu_matchedjet_isolation2",
                                     label=r"Leading-Mu-LJ Matched Jet Isolation2"),
                   lambda objs, mask: abs(derived_objs["leading_matched_jets"](objs,0.4)[mask,0].energy - objs["ljs"][:,0:1][mask].energy)
                       / objs["ljs"][:,0:1][mask].energy),
        ],
        evt_mask=lambda objs: ak.num(abs(objs["ljs"][:,0:1].phi - objs["ljs"][:,1:2].phi)) == ak.num(derived_objs["leading_mu_matched_jets"](objs,0.4)),
    ),
    
    "lj_lj_absdphi_invmass": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 2*math.pi, name="|$\Delta\phi$| ($LJ_{0}$, $LJ_{1}$)"),
                   lambda objs, mask: abs(objs["ljs"][mask, 1].phi - objs["ljs"][mask, 0].phi)),
            h.Axis(hist.axis.Regular(100, 0, 1200, name="ljlj_mass",
                                     label=r"Invariant Mass ($LJ_{0}$, $LJ_{1}$)"),
                   lambda objs, mask: objs["ljs"][mask, :2].sum().mass),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 1,
    ),
    "lj_lj_absdphi_absdR": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 2*math.pi, name="|$\Delta\phi$| ($LJ_{0}$, $LJ_{1}$)"),
                   lambda objs, mask: abs(objs["ljs"][mask, 1].phi - objs["ljs"][mask, 0].phi)),
            h.Axis(hist.axis.Regular(100, 0, 6, name="|$\Delta$R| ($LJ_{0}$, $LJ_{1}$)"),
                   lambda objs, mask: objs["ljs"][mask, 1].delta_r(objs["ljs"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 1,
    ),
    "lj_lj_absdphi_absdeta": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 2*math.pi, name="|$\Delta\phi$| ($LJ_{0}$, $LJ_{1}$)"),
                   lambda objs, mask: abs(objs["ljs"][mask, 1].phi - objs["ljs"][mask, 0].phi)),
            h.Axis(hist.axis.Regular(100, 0, 6, name="|$\Delta\eta$| ($LJ_{0}$, $LJ_{1}$)"),
                   lambda objs, mask: abs(objs["ljs"][mask, 1].eta - objs["ljs"][mask, 0].eta)),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 1,
    ),
    "lj_lj_absdphi_ptRatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 2*math.pi, name="|$\Delta\phi$| ($LJ_{0}$, $LJ_{1}$)"),
                   lambda objs, mask: abs(objs["ljs"][mask, 1].phi - objs["ljs"][mask, 0].phi)),
            h.Axis(hist.axis.Regular(100, 1.0, 2.0, name="lj_lj_ptRatio",
                                     label="Leading LJ PT / Subleading LJ PT"),
                   lambda objs, mask: objs["ljs"][mask, 0].pt / objs["ljs"][mask, 1].pt),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 1,
    ),
    "lj_lj_absdR_absdeta": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 6, name="|$\Delta$R| ($LJ_{0}$, $LJ_{1}$)"),
                   lambda objs, mask: objs["ljs"][mask, 1].delta_r(objs["ljs"][mask, 0])),
            h.Axis(hist.axis.Regular(100, 0, 6, name="|$\Delta\eta$| ($LJ_{0}$, $LJ_{1}$)"),
                   lambda objs, mask: abs(objs["ljs"][mask, 1].eta - objs["ljs"][mask, 0].eta)),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 1,
    ),
    "lj_lj_absdR_invmass": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 6, name="|$\Delta$R| ($LJ_{0}$, $LJ_{1}$)"),
                   lambda objs, mask: objs["ljs"][mask, 1].delta_r(objs["ljs"][mask, 0])),
            h.Axis(hist.axis.Regular(100, 0, 1200, name="ljlj_mass",
                                     label=r"Invariant Mass ($LJ_{0}$, $LJ_{1}$)"),
                   lambda objs, mask: objs["ljs"][mask, :2].sum().mass),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 1,
    ),
    "lj_lj_absdR_ptRatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 6, name="|$\Delta$R| ($LJ_{0}$, $LJ_{1}$)"),
                   lambda objs, mask: objs["ljs"][mask, 1].delta_r(objs["ljs"][mask, 0])),
            h.Axis(hist.axis.Regular(100, 1.0, 2.0, name="lj_lj_ptRatio",
                   label="Leading LJ PT / Subleading LJ PT"),
                   lambda objs, mask: objs["ljs"][mask, 0].pt / objs["ljs"][mask, 1].pt),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 1,
    ),
    "lj_lj_absdeta_invmass": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 6, name="|$\Delta\eta$| ($LJ_{0}$, $LJ_{1}$)"),
                   lambda objs, mask: abs(objs["ljs"][mask, 1].eta - objs["ljs"][mask, 0].eta)),
            h.Axis(hist.axis.Regular(100, 0, 1200, name="ljlj_mass",
                                     label=r"Invariant Mass ($LJ_{0}$, $LJ_{1}$)"),
                   lambda objs, mask: objs["ljs"][mask, :2].sum().mass),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 1,
    ),
    "lj_lj_absdeta_ptRatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 6, name="|$\Delta\eta$| ($LJ_{0}$, $LJ_{1}$)"),
                   lambda objs, mask: abs(objs["ljs"][mask, 1].eta - objs["ljs"][mask, 0].eta)),
            h.Axis(hist.axis.Regular(100, 1.0, 2.0, name="lj_lj_ptRatio",
                   label="Leading LJ PT / Subleading LJ PT"),
                   lambda objs, mask: objs["ljs"][mask, 0].pt / objs["ljs"][mask, 1].pt),
        ],
        evt_mask=lambda objs: ak.num(objs["ljs"]) > 1,
    ),
    # gen
    "gen_abspid": h.Histogram(
        [
            h.Axis(hist.axis.Integer(0, 40, name="gen_abspid"),
                   lambda objs, mask: abs(objs["gens"].pdgId)),
        ],
    ),
    # genelectron
    "genE_n": obj_attr("genEs", "n"),
    "genE_pt": obj_attr("genEs", "pt"),
    "genE_pt_highRange": obj_attr("genEs", "pt", xmax=700),
    "genE0_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 200, name="genE0_pt",
                                     label=r"Leading gen-level electron $p_{T}$ [GeV]"),
                   lambda objs, mask: objs["genEs"][mask, 0].pt),
        ],
        evt_mask=lambda objs: ak.num(objs["genEs"]) > 0,
    ),
    "genE0_pt_highRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(70, 0, 700, name="genE_pt",
                                     label=r"Leading gen-level electron $p_{T}$ [GeV]"),
                   lambda objs, mask: objs["genEs"][mask, 0].pt),
        ],
        evt_mask=lambda objs: ak.num(objs["genEs"]) > 0,
    ),
    "genE1_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 200, name="genE1_pt",
                                     label=r"Sub-leading gen-level electron $p_{T}$ [GeV]"),
                   lambda objs, mask: objs["genEs"][mask, 1].pt),
        ],
        evt_mask=lambda objs: ak.num(objs["genEs"]) > 1,
    ),
    "genE1_pt_highRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(70, 0, 700, name="genE_pt",
                                     label=r"Sub-leading gen-level electron $p_{T}$ [GeV]"),
                   lambda objs, mask: objs["genEs"][mask, 1].pt),
        ],
        evt_mask=lambda objs: ak.num(objs["genEs"]) > 1,
    ),
    "genE_eta_phi": obj_eta_phi("genEs"),
    "genE_parent_absPdgId": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 50, name="genE_parent_absPdgId"),
                   lambda objs, mask: abs(objs["genEs"].parent.pdgId)),
        ],
    ),
    # genelectron-genelectron
    "genE_genE_dR": h.Histogram(
        [
            # dR(subleading gen E, leading gen E)
            h.Axis(hist.axis.Regular(100, 0, 1.0, name="genE_genE_dR",
                                     label=r"$\Delta R$($e_0^{gen}$, $e_1^{gen}$)"),
                   lambda objs, mask: objs["genEs"][mask, 1].delta_r(objs["genEs"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genEs"]) > 1,
    ),
    "genE_genE_dR_lowRange": h.Histogram(
        [
            # dR(subleading gen E, leading gen E)
            h.Axis(hist.axis.Regular(75, 0, 0.5, name="genE_genE_dR_lowRange",
                                     label=r"$\Delta R$($e_0^{gen}$, $e_1^{gen}$)"),
                   lambda objs, mask: objs["genEs"][mask, 1].delta_r(objs["genEs"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genEs"]) > 1,
    ),
    "genE_genE_dR_XLowRange": h.Histogram(
        [
            # dR(subleading gen E, leading gen E)
            h.Axis(hist.axis.Regular(50, 0, 0.1, name="genE_genE_dR_lowRange",
                                     label=r"$\Delta R$($e_0^{gen}$, $e_1^{gen}$)"),
                   lambda objs, mask: objs["genEs"][mask, 1].delta_r(objs["genEs"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genEs"]) > 1,
    ),
    "genE_genE_dR_XXLowRange": h.Histogram(
        [
            # dR(subleading gen E, leading gen E)
            h.Axis(hist.axis.Regular(50, 0, 0.04, name="genE_genE_dR_lowRange",
                                     label=r"$\Delta R$($e_0^{gen}$, $e_1^{gen}$)"),
                   lambda objs, mask: objs["genEs"][mask, 1].delta_r(objs["genEs"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genEs"]) > 1,
    ),
    "genE_genE_dEta": h.Histogram(
        [
            # abs(dEta(subleading gen E, leading gen E))
            h.Axis(hist.axis.Regular(50, 0, 1.0, name="genE_genE_dEta",
                                     label=r"$\Delta\, \eta$($e_0^{gen}$, $e_1^{gen}$)"),
                   lambda objs, mask: abs(objs["genEs"][mask, 1].eta
                                          - objs["genEs"][mask, 0].eta)),
        ],
        evt_mask=lambda objs: ak.num(objs["genEs"]) > 1,
    ),
    "genE_genE_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 200, name="genE_genE_pt"),
                   lambda objs, mask: objs["genEs"][mask, :2].sum().pt),
        ],
        evt_mask=lambda objs: ak.num(objs["genEs"]) > 1,
    ),
    # genmuon
    "genMu_n": obj_attr("genMus", "n"),
    "genMu_pt": obj_attr("genMus", "pt"),
    "genMu_pt_highRange": obj_attr("genMus", "pt", xmax=700),
    "genMu0_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(200, 0, 200, name="genMu0_pt",
                                     label=r"Leading gen-level muon $p_{T}$ [GeV]"),
                   lambda objs, mask: objs["genMus"][mask, 0].pt),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 0,
    ),
    "genMu0_pt_highRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(70, 0, 700, name="genMu0_pt",
                                     label=r"Leading gen-level muon $p_{T}$ [GeV]"),
                   lambda objs, mask: objs["genMus"][mask, 0].pt),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 0,
    ),
    "genMu1_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 200, name="genMu1_pt",
                                     label=r"Sub-leading gen-level muon $p_{T}$ [GeV]"),
                   lambda objs, mask: objs["genMus"][mask, 1].pt),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    "genMu1_pt_highRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(70, 0, 700, name="genMu1_pt",
                                     label=r"Sub-leading gen-level muon $p_{T}$ [GeV]"),
                   lambda objs, mask: objs["genMus"][mask, 1].pt),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    "genMu_eta_phi": obj_eta_phi("genMus"),
    "genMu_parent_absPdgId": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 50, name="genMu_parent_absPdgId"),
                   lambda objs, mask: abs(objs["genMus"].parent.pdgId)),
        ],
    ),
    # genmuon-genmuon
    "genMu_genMu_dR": h.Histogram(
        [
            # dR(subleading gen Mu, leading gen Mu)
            h.Axis(hist.axis.Regular(50, 0, 1.0, name="genMu_genMu_dR",
                                     label=r"$\Delta R$($\mu_0^{gen}$, $\mu_1^{gen}$)"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(
                       objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    "genMu_genMu_dR_lowRange": h.Histogram(
        [
            # dR(subleading gen Mu, leading gen Mu)
            h.Axis(hist.axis.Regular(100, 0, 0.5, name="genMu_genMu_dR_lowRange",
                                     label=r"$\Delta R$($\mu_0^{gen}$, $\mu_1^{gen}$)"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(
                       objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    "genMu_genMu_dR_XLowRange": h.Histogram(
        [
            # dR(subleading gen Mu, leading gen Mu)
            h.Axis(hist.axis.Regular(100, 0, 0.1, name="genMu_genMu_dR_lowRange",
                                     label=r"$\Delta R$($\mu_0^{gen}$, $\mu_1^{gen}$)"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(
                       objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    "genMu_genMu_dR_XXLowRange": h.Histogram(
        [
            # dR(subleading gen Mu, leading gen Mu)
            h.Axis(hist.axis.Regular(100, 0, 0.04, name="genMu_genMu_dR_lowRange",
                                     label=r"$\Delta R$($\mu_0^{gen}$, $\mu_1^{gen}$)"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(
                       objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    "genMu_genMu_dEta": h.Histogram(
        [
            # abs(dEta(subleading gen Mu, leading gen Mu))
            h.Axis(hist.axis.Regular(50, 0, 1.0, name="genMu_genMu_dEta",
                                     label=r"$\Delta\, \eta$($\mu_0^{gen}$, $\mu_1^{gen}$)"),
                   lambda objs, mask: abs(objs["genMus"][mask, 1].eta
                                          - objs["genMus"][mask, 0].eta)),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    "genMu_genMu_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 200, name="genMu_genMu_pt"),
                   lambda objs, mask: objs["genMus"][mask, :2].sum().pt),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    #dsamuon-genAs_toMu
    "dsamuon_absd0_genAs_toMu_lxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 200, name="dsamuon_absd0",
                                     label=r"dsa muon $|d_0|$ [cm]"),
                   lambda objs, mask: abs(objs["dsaMuons"].d0)),
                   #added the function ak.ones_like to match delta r array with the d0 array.
            h.Axis(hist.axis.Regular(25, 0,400, name="genA_lxy"),
                   #Added the function ak.ones_like to match delta R array with the d0 array.
                   lambda objs, mask: lxy(objs["genAs_toMu"])[:,0]*ak.ones_like(objs["dsaMuons"].d0)),
        ],
        #evt_mask=lambda objs: ak.num(objs["genAs_toMu"]) > 0,
    ),
    #dsamuon-genmuon
    "dsaMuon_absD0_genMus_dR": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 200, name="dsaMuon_absD0",
                                     label=r"DSA muon $|d_0|$ [cm]"),
                   lambda objs, mask: abs(objs["dsaMuons"].d0)),
            # dR(subleading gen Mu, leading gen Mu)
            h.Axis(hist.axis.Regular(25, 0, 0.4, name="genMu_genMu_dR",
                                     label=r"$\Delta R$($\mu_0^{gen}$, $\mu_1^{gen}$)"),
                   #Added the function ak.ones_like to match delta R array with the d0 array.
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(
                       objs["genMus"][mask, 0])*ak.ones_like(objs["dsaMuons"].d0)),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1
    ),
    "leadingDsaMuon_absD0_genMus_dR": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 200, name="dsaMuon_absD0",
                                     label=r"Leading DSA muon $|d_0|$ [cm]"),
                   lambda objs, mask: abs(objs["dsaMuons"][mask, 0].d0)),
            # dR(subleading gen Mu, leading gen Mu)
            h.Axis(hist.axis.Regular(25, 0, 0.4, name="genMu_genMu_dR",
                                     label=r"$\Delta R$($\mu_0^{gen}$, $\mu_1^{gen}$)"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ((ak.num(objs["genMus"]) > 1) & (ak.num(objs["dsaMuons"]) > 0))
    ),
    "subLeadingDsaMuon_absD0_genMus_dR": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 200, name="dsaMuon_absD0",
                                     label=r"Sub Leading DSA muon $|d_0|$ [cm]"),
                   lambda objs, mask: abs(objs["dsaMuons"][mask, 1].d0)),
            # dR(subleading gen Mu, leading gen Mu)
            h.Axis(hist.axis.Regular(25, 0, 0.4, name="genMu_genMu_dR",
                                     label=r"$\Delta R$($\mu_0^{gen}$, $\mu_1^{gen}$)"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ((ak.num(objs["genMus"]) > 1) & (ak.num(objs["dsaMuons"]) > 1))
    ),
    "dsaMuon_absD0_genMus_dR_XLowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 200, name="dsaMuon_absD0",
                                     label=r"DSA muon $|d_0|$ [cm]"),
                   lambda objs, mask: abs(objs["dsaMuons"].d0)),
            # dR(subleading gen Mu, leading gen Mu)
            h.Axis(hist.axis.Regular(25, 0, 0.1, name="genMu_genMu_dR",
                                     label=r"$\Delta R$($\mu_0^{gen}$, $\mu_1^{gen}$)"),
                   #Added the function ak.ones_like to match delta R array with the d0 array.
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(
                       objs["genMus"][mask, 0])*ak.ones_like(objs["dsaMuons"].d0)),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1
    ),
    "leadingDsaMuon_absD0_genMus_dR_XLowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 200, name="dsaMuon_absD0",
                                     label=r"Leading DSA muon $|d_0|$ [cm]"),
                   lambda objs, mask: abs(objs["dsaMuons"][mask, 0].d0)),
            # dR(subleading gen Mu, leading gen Mu)
            h.Axis(hist.axis.Regular(25, 0, 0.1, name="genMu_genMu_dR",
                                     label=r"$\Delta R$($\mu_0^{gen}$, $\mu_1^{gen}$)"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ((ak.num(objs["genMus"]) > 1) & (ak.num(objs["dsaMuons"]) > 0))
    ),
    "subLeadingDsaMuon_absD0_genMus_dR_XLowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 200, name="dsaMuon_absD0",
                                     label=r"Sub Leading DSA muon $|d_0|$ [cm]"),
                   lambda objs, mask: abs(objs["dsaMuons"][mask, 1].d0)),
            # dR(subleading gen Mu, leading gen Mu)
            h.Axis(hist.axis.Regular(25, 0, 0.1, name="genMu_genMu_dR",
                                     label=r"$\Delta R$($\mu_0^{gen}$, $\mu_1^{gen}$)"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ((ak.num(objs["genMus"]) > 1) & (ak.num(objs["dsaMuons"]) > 1))
    ),
    "dsaMuon_absD0_genMus_dR_XXLowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 200, name="dsaMuon_absD0",
                                     label=r"DSA muon $|d_0|$ [cm]"),
                   lambda objs, mask: abs(objs["dsaMuons"].d0)),
            # dR(subleading gen Mu, leading gen Mu)
            h.Axis(hist.axis.Regular(25, 0, 0.03, name="genMu_genMu_dR",
                                     label=r"$\Delta R$($\mu_0^{gen}$, $\mu_1^{gen}$)"),
                   #Added the function ak.ones_like to match delta R array with the d0 array.
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(
                       objs["genMus"][mask, 0])*ak.ones_like(objs["dsaMuons"].d0)),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1
    ),
    "leadingDsaMuon_absD0_genMus_dR_XXLowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 200, name="dsaMuon_absD0",
                                     label=r"Leading DSA muon $|d_0|$ [cm]"),
                   lambda objs, mask: abs(objs["dsaMuons"][mask, 0].d0)),
            # dR(subleading gen Mu, leading gen Mu)
            h.Axis(hist.axis.Regular(25, 0, 0.03, name="genMu_genMu_dR",
                                     label=r"$\Delta R$($\mu_0^{gen}$, $\mu_1^{gen}$)"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ((ak.num(objs["genMus"]) > 1) & (ak.num(objs["dsaMuons"]) > 0))
    ),
    "subLeadingDsaMuon_absD0_genMus_dR_XXLowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 200, name="dsaMuon_absD0",
                                     label=r"Sub Leading DSA muon $|d_0|$ [cm]"),
                   lambda objs, mask: abs(objs["dsaMuons"][mask, 1].d0)),
            # dR(subleading gen Mu, leading gen Mu)
            h.Axis(hist.axis.Regular(25, 0, 0.03, name="genMu_genMu_dR",
                                     label=r"$\Delta R$($\mu_0^{gen}$, $\mu_1^{gen}$)"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ((ak.num(objs["genMus"]) > 1) & (ak.num(objs["dsaMuons"]) > 1))
    ),
    # dsamuon-genmuon, dR 0.4 window
    "dsaMuon_genMu_ptRatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(200, 0, 2.0, name="dsaMuon_genMu_ptRatio"),
                   lambda objs, mask: objs["dsaMuons"].pt
                       / objs["dsaMuons"].nearest(objs["genMus"], threshold=0.4).pt),
        ],
    ),
    "dsaMuon0_genMu_ptRatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(200, 0, 2.0, name="dsaMuon0_genMu_ptRatio"),
                   lambda objs, mask: (objs["dsaMuons"][mask, 0:1].pt
                       / objs["dsaMuons"][mask, 0:1].nearest(objs["genMus"][mask], threshold=0.4).pt)),
        ],
        evt_mask=lambda objs: ak.num(matched(objs["dsaMuons"] ,objs["genMus"], 0.4)) > 0,
    ),
    "dsaMuon1_genMu_ptRatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(200, 0, 2.0, name="dsaMuon1_genMu_ptRatio"),
                   lambda objs, mask: (objs["dsaMuons"][mask, 1:2].pt
                       / objs["dsaMuons"][mask, 1:2].nearest(objs["genMus"][mask], threshold=0.4).pt)),
        ],
        evt_mask=lambda objs: ak.num(matched(objs["dsaMuons"], objs["genMus"], 0.4)) > 1,
    ),
    # pfmuon-genmuon, dR 0.4 window
    "pfMuon_genMu_ptRatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(200, 0, 2.0, name="pfMuon_genMu_ptRatio"),
                   lambda objs, mask: objs["muons"].pt
                       / objs["muons"].nearest(objs["genMus"], threshold=0.4).pt),
        ],
    ),
    "pfMuon0_genMu_ptRatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(200, 0, 2.0, name="pfMuon0_genMu_ptRatio"),
                   lambda objs, mask: (objs["muons"][mask,0:1].pt
                       / objs["muons"][mask,0:1].nearest(objs["genMus"][mask], threshold=0.4).pt)),
        ],
        evt_mask=lambda objs: ak.num(matched(objs["muons"], objs["genMus"], 0.4)) > 0,
    ),
    "pfMuon1_genMu_ptRatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(200, 0, 2.0, name="pfMuon1_genMu_ptRatio"),
                   lambda objs, mask: (objs["muons"][mask,1:2].pt
                       / objs["muons"][mask,1:2].nearest(objs["genMus"][mask], threshold=0.4).pt)),
        ],
        evt_mask=lambda objs: ak.num(matched(objs["muons"], objs["genMus"], 0.4)) > 1,
    ),
    # gen dark photons (A)
    "genAs_n": obj_attr("genAs", "n"),
    "genAs_toMu_n": obj_attr("genAs_toMu", "n"),
    "genAs_toE_n": obj_attr("genAs_toE", "n"),
    "genAs_pt": obj_attr("genAs", "pt", xmax=200),
    "genAs_pt_highRange": obj_attr("genAs", "pt", xmax=700),
    "genAs_eta": obj_attr("genAs", "eta", xmin=-3, xmax=3),
    "genAs_eta_phi": obj_eta_phi("genAs"),
    "genAs_toMu_matched_muLj_n": h.Histogram(
        [
            h.Axis(hist.axis.Regular(10, 0, 10, name="genA_n"),
                   lambda objs, mask: ak.num(derived_objs["genAs_toMu_matched_muLj"](objs, 0.4)))
        ],
    ),
    "genAs_toE_matched_egmLj_n": h.Histogram(
        [
            h.Axis(hist.axis.Regular(10, 0, 10, name="genA_n"),
                   lambda objs, mask: ak.num(derived_objs["genAs_toE_matched_egmLj"](objs, 0.4)))
        ],
    ),
    "genAs_x_y": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0.000, 0.025, name="genAs_x"),
                   lambda objs, mask: objs["genAs"].vx),
            h.Axis(hist.axis.Regular(100, 0.025, 0.050, name="genAs_y"),
                   lambda objs, mask: objs["genAs"].vy),
        ],
    ),
    "genAs_children_x_y": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, -200, 200, name="genAs_children_x"),
                   lambda objs, mask: objs["genAs"].children.vx),
            h.Axis(hist.axis.Regular(100, -200, 200, name="genAs_children_y"),
                   lambda objs, mask: objs["genAs"].children.vy),
        ],
    ),
    "genAs_lxy": obj_attr("genAs", "lxy", xmax=500),
    "genAs_lxy_lowRange": obj_attr("genAs", "lxy", xmax=10),
    "genAs_children_n": h.Histogram(
        [
            h.Axis(hist.axis.Regular(10, 0, 10, name="genAs_children_n"),
                   lambda objs, mask: ak.num(objs["genAs"].children)),
        ],
    ),
    "genAs_children_absPdgId": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 50, name="genAs_children_absPdgId"),
                   lambda objs, mask: abs(objs["genAs"].children.pdgId)),
        ],
    ),
    "genAs_toMu_lxy": obj_attr("genAs_toMu", "lxy", xmax=500),
    "genAs_toMu_pt": obj_attr("genAs_toMu", "pt", xmax=200),
    "genAs_toMu_pt_highRange": obj_attr("genAs_toMu", "pt", xmax=700),
    "genAs_toMu_eta": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, -3, 3, name=r"$Z_d$ $\eta$"),
                   lambda objs, mask: objs["genAs_toMu"].eta ),
        ],
    ),
    "genAs_toE_lxy": obj_attr("genAs_toE", "lxy", xmax=500),
    "genAs_toE_lxy_lowRange": obj_attr("genAs_toE", "lxy", xmax=20),
    "genAs_toE_lxy_midRange": obj_attr("genAs_toE", "lxy", xmin=40, xmax=80),
    "genAs_toE_lxy_ecal": obj_attr("genAs_toE", "lxy", xmin=125, xmax=135),
    "genAs_toE_pt": obj_attr("genAs_toE", "pt", xmax=200),
    "genAs_toE_pt_highRange": obj_attr("genAs_toE", "pt", xmax=700),
    "genAs_toE_eta": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, -3, 3, name=r"$Z_d$ $\eta$"),
                   lambda objs, mask: objs["genAs_toE"].eta ),
        ],
    ),
    "genAs_matched_lj_lxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 500, name=r"$Z_d$ $L_{xy}$ $(cm)$"),
                   lambda objs, mask: lxy(derived_objs["genAs_matched_lj"](objs, 0.4)) ),
        ],
    ),
    "genAs_toMu_matched_lj_lxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 500, name=r"$Z_d$ $L_{xy}$ $(cm)$"),
                   lambda objs, mask: lxy(derived_objs["genAs_toMu_matched_lj"](objs, 0.4)) ),
        ],
    ),
    "genAs_toE_matched_lj_lxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 500, name=r"$Z_d$ $L_{xy}$ $(cm)$"),
                   lambda objs, mask: lxy(derived_objs["genAs_toE_matched_lj"](objs, 0.4)) ),
        ],
    ),
    "genAs_matched_muLj_lxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 500, name=r"$Z_d$ $L_{xy}$ $(cm)$"),
                   lambda objs, mask: lxy(derived_objs["genAs_matched_muLj"](objs, 0.4)) ),
        ],
    ),
    "genAs_toMu_matched_muLj_lxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 500, name=r"$Z_d$ $L_{xy}$ $(cm)$"),
                   lambda objs, mask: lxy(derived_objs["genAs_toMu_matched_muLj"](objs, 0.4)) ),
        ],
    ),
    "genAs_toMu_matched_muLj_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 200, name=r"$Z_d$ $p_T$ $(GeV)$"),
                   lambda objs, mask: abs(derived_objs["genAs_toMu_matched_muLj"](objs, 0.4).pt) ),
        ],
    ),
    "genAs_toMu_matched_muLj_pt_highRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(140, 0, 700, name=r"$Z_d$ $p_T$ $(GeV)$"),
                   lambda objs, mask: abs(derived_objs["genAs_toMu_matched_muLj"](objs, 0.4).pt) ),
        ],
    ),
    "genAs_toMu_matched_muLj_eta": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, -3, 3, name=r"$Z_d$ $\eta$"),
                   lambda objs, mask: derived_objs["genAs_toMu_matched_muLj"](objs, 0.4).eta ),
        ],
    ),
    "genAs_matched_egmLj_lxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 500, name=r"$Z_d$ $L_{xy}$ $(cm)$"),
                   lambda objs, mask: lxy(derived_objs["genAs_matched_egmLj"](objs, 0.4)) ),
        ],
    ),
    "genAs_toE_matched_egmLj_lxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 500, name=r"$Z_d$ $L_{xy}$ $(cm)$"),
                   lambda objs, mask: lxy(derived_objs["genAs_toE_matched_egmLj"](objs, 0.4)) ),
        ],
    ),
    "genAs_toE_matched_egmLj_lxy_lowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 20, name=r"$Z_d$ $L_{xy}$ $(cm)$"),
                   lambda objs, mask: lxy(derived_objs["genAs_toE_matched_egmLj"](objs, 0.4)) ),
        ],
    ),
    "genAs_toE_matched_egmLj_lxy_midRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 40, 80, name=r"$Z_d$ $L_{xy}$ $(cm)$"),
                   lambda objs, mask: lxy(derived_objs["genAs_toE_matched_egmLj"](objs, 0.4)) ),
        ],
    ),
    "genAs_toE_matched_egmLj_lxy_ecal": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 125, 135, name=r"$Z_d$ $L_{xy}$ $(cm)$"),
                   lambda objs, mask: lxy(derived_objs["genAs_toE_matched_egmLj"](objs, 0.4)) ),
        ],
    ),
    "genAs_matched_lj_n": h.Histogram(
        [
            h.Axis(hist.axis.Regular(10, 0, 10, name="genAs_matched_lj_n"),
                   lambda objs, mask: ak.num(derived_objs["genAs_matched_lj"](objs, 0.4)) ),
        ],
    ),
    "genAs_matched_lj_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 200, name=r"$Z_d$ $p_T$ $(GeV)$"),
                   lambda objs, mask: abs(derived_objs["genAs_matched_lj"](objs, 0.4).pt) ),
        ],
    ),
    "genAs_matched_lj_pt_highRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(140, 0, 700, name=r"$Z_d$ $p_T$ $(GeV)$"),
                   lambda objs, mask: abs(derived_objs["genAs_matched_lj"](objs, 0.4).pt) ),
        ],
    ),
    "genAs_matched_lj_eta": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, -3, 3, name=r"$Z_d$ $\eta$"),
                   lambda objs, mask: derived_objs["genAs_matched_lj"](objs, 0.4).eta ),
        ],
    ),
    "genAs_toE_matched_egmLj_pt": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 200, name=r"$Z_d$ $p_T$ $(GeV)$"),
                   lambda objs, mask: abs(derived_objs["genAs_toE_matched_egmLj"](objs, 0.4).pt) ),
        ],
    ),
    "genAs_toE_matched_egmLj_pt_highRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(140, 0, 700, name=r"$Z_d$ $p_T$ $(GeV)$"),
                   lambda objs, mask: abs(derived_objs["genAs_toE_matched_egmLj"](objs, 0.4).pt) ),
        ],
    ),
    "genAs_toE_matched_egmLj_eta": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, -3, 3, name=r"$Z_d$ $\eta$"),
                   lambda objs, mask: derived_objs["genAs_toE_matched_egmLj"](objs, 0.4).eta ),
        ],
    ),
    "genAs_pt_lxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 200, name="genAs_pt"),
                   lambda objs, mask: abs(objs["genAs"].pt)),
            h.Axis(hist.axis.Regular(250, 0, 500, name="genAs_lxy"),
                   lambda objs, mask: lxy(objs["genAs"])),
        ],
    ),
    "genMu0_pt_MuMudR": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0, 200, name="genMu0_pt",
                                     label=r"Leading gen-level muon $p_{T}$ [GeV]"),
                   lambda objs, mask: objs["genMus"][mask, 0].pt),
            h.Axis(hist.axis.Regular(25, 0, 0.4, name="genMu_genMu_dR_lowRange"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    "genMu0_pt_MuMudR_XLowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0, 200, name="genMu0_pt",
                                     label=r"Leading gen-level muon $p_{T}$ [GeV]"),
                   lambda objs, mask: objs["genMus"][mask, 0].pt),
            h.Axis(hist.axis.Regular(25, 0, 0.1, name="genMu_genMu_dR_lowRange"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    "genMu0_pt_MuMudR_XXLowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0, 200, name="genMu0_pt",
                                     label=r"Leading gen-level muon $p_{T}$ [GeV]"),
                   lambda objs, mask: objs["genMus"][mask, 0].pt),
            h.Axis(hist.axis.Regular(25, 0, 0.04, name="genMu_genMu_dR_lowRange"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    "genMu0_pt_highRange_MuMudR": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0, 700, name="genMu0_pt",
                                     label=r"Sub-Leading gen-level muon $p_{T}$ [GeV]"),
                   lambda objs, mask: objs["genMus"][mask, 0].pt),
            h.Axis(hist.axis.Regular(20, 0, 0.25, name="genMu_genMu_dR_lowRange"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    "genMu0_pt_highRange_MuMudR_XLowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0, 700, name="genMu0_pt",
                                     label=r"Sub-Leading gen-level muon $p_{T}$ [GeV]"),
                   lambda objs, mask: objs["genMus"][mask, 0].pt),
            h.Axis(hist.axis.Regular(20, 0, 0.06, name="genMu_genMu_dR_lowRange"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    "genMu0_pt_highRange_MuMudR_XXLowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0, 700, name="genMu0_pt",
                                     label=r"Sub-Leading gen-level muon $p_{T}$ [GeV]"),
                   lambda objs, mask: objs["genMus"][mask, 0].pt),
            h.Axis(hist.axis.Regular(10, 0, 0.01, name="genMu_genMu_dR_lowRange"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    "genMu1_pt_MuMudR": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0, 200, name="genMu0_pt",
                                     label=r"Sub-Leading gen-level muon $p_{T}$ [GeV]"),
                   lambda objs, mask: objs["genMus"][mask, 1].pt),
            h.Axis(hist.axis.Regular(25, 0, 0.4, name="genMu_genMu_dR_lowRange"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    "genMu1_pt_MuMudR_XLowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0, 200, name="genMu0_pt",
                                     label=r"Sub-Leading gen-level muon $p_{T}$ [GeV]"),
                   lambda objs, mask: objs["genMus"][mask, 1].pt),
            h.Axis(hist.axis.Regular(25, 0, 0.1, name="genMu_genMu_dR_lowRange"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    "genMu1_pt_MuMudR_XXLowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0, 200, name="genMu0_pt",
                                     label=r"Sub-Leading gen-level muon $p_{T}$ [GeV]"),
                   lambda objs, mask: objs["genMus"][mask, 1].pt),
            h.Axis(hist.axis.Regular(25, 0, 0.04, name="genMu_genMu_dR_lowRange"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    "genMu1_pt_highRange_MuMudR": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0, 700, name="genMu0_pt",
                                     label=r"Sub-Leading gen-level muon $p_{T}$ [GeV]"),
                   lambda objs, mask: objs["genMus"][mask, 1].pt),
            h.Axis(hist.axis.Regular(20, 0, 0.25, name="genMu_genMu_dR_lowRange"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    "genMu1_pt_highRange_MuMudR_XLowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0, 700, name="genMu0_pt",
                                     label=r"Sub-Leading gen-level muon $p_{T}$ [GeV]"),
                   lambda objs, mask: objs["genMus"][mask, 1].pt),
            h.Axis(hist.axis.Regular(20, 0, 0.06, name="genMu_genMu_dR_lowRange"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    "genMu1_pt_highRange_MuMudR_XXLowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0, 700, name="genMu0_pt",
                                     label=r"Sub-Leading gen-level muon $p_{T}$ [GeV]"),
                   lambda objs, mask: objs["genMus"][mask, 1].pt),
            h.Axis(hist.axis.Regular(10, 0, 0.01, name="genMu_genMu_dR_lowRange"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    "genAs_toMu_pt_MuMudR_lowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0,200, name="genAs_pt"),
                   lambda objs, mask: abs(objs["genAs_toMu"].pt)),
            h.Axis(hist.axis.Regular(25, 0, 0.4, name="genMu_genMu_dR_lowRange"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    "genAs_toMu_pt_MuMudR_XLowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0,200, name="genAs_pt"),
                   lambda objs, mask: abs(objs["genAs_toMu"].pt)),
            h.Axis(hist.axis.Regular(25, 0, 0.1, name="genMu_genMu_dR_lowRange"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    "genAs_toMu_pt_MuMudR_XXLowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0,200, name="genAs_pt"),
                   lambda objs, mask: abs(objs["genAs_toMu"].pt)),
            h.Axis(hist.axis.Regular(25, 0, 0.04, name="genMu_genMu_dR_lowRange"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    "genAs_toMu_lxy_MuMudR": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0,400, name="genAs_lxy"),
                   lambda objs, mask: lxy(objs["genAs_toMu"])),
            h.Axis(hist.axis.Regular(25, 0, 0.4, name="genMu_genMu_dR_lowRange"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    "genAs_toMu_lxy_MuMudR_XLowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0,400, name="genAs_lxy"),
                   lambda objs, mask: lxy(objs["genAs_toMu"])),
            h.Axis(hist.axis.Regular(25, 0, 0.1, name="genMu_genMu_dR_lowRange"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    "genAs_toMu_lxy_MuMudR_XXLowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0,400, name="genAs_lxy"),
                   lambda objs, mask: lxy(objs["genAs_toMu"])),
            h.Axis(hist.axis.Regular(25, 0, 0.04, name="genMu_genMu_dR_lowRange"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    "genAs_toMu_lxy_pt_lowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0, 400, name="genAs_lxy"),
                   lambda objs, mask: lxy(objs["genAs_toMu"])),
            h.Axis(hist.axis.Regular(25, 0,200, name="genAs_pt"),
                   lambda objs, mask: abs(objs["genAs_toMu"].pt)),
        ],
    ),
    "genAs_toE_pt_EEdR_lowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0,200, name="genAs_pt"),
                   lambda objs, mask: abs(objs["genAs_toE"].pt)),
            h.Axis(hist.axis.Regular(25, 0, 0.4, name="genE_genE_dR_lowRange"),
                   lambda objs, mask: objs["genEs"][mask, 1].delta_r(objs["genEs"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genEs"]) > 1,
    ),
    "genAs_toE_pt_EEdR_XLowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0,200, name="genAs_pt"),
                   lambda objs, mask: abs(objs["genAs_toE"].pt)),
            h.Axis(hist.axis.Regular(25, 0, 0.1, name="genE_genE_dR_lowRange"),
                   lambda objs, mask: objs["genEs"][mask, 1].delta_r(objs["genEs"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genEs"]) > 1,
    ),
    "genAs_toE_pt_EEdR_XXLowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0,200, name="genAs_pt"),
                   lambda objs, mask: abs(objs["genAs_toE"].pt)),
            h.Axis(hist.axis.Regular(25, 0, 0.04, name="genE_genE_dR_lowRange"),
                   lambda objs, mask: objs["genEs"][mask, 1].delta_r(objs["genEs"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genEs"]) > 1,
    ),
    "genAs_toE_lxy_EEdR": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0,150, name="genAs_lxy"),
                   lambda objs, mask: lxy(objs["genAs_toE"])),
            h.Axis(hist.axis.Regular(25, 0, 0.4, name="genE_genE_dR_lowRange"),
                   lambda objs, mask: objs["genEs"][mask, 1].delta_r(objs["genEs"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genEs"]) > 1,
    ),
    "genAs_toE_lxy_EEdR_XLowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0,150, name="genAs_lxy"),
                   lambda objs, mask: lxy(objs["genAs_toE"])),
            h.Axis(hist.axis.Regular(25, 0, 0.1, name="genE_genE_dR_lowRange"),
                   lambda objs, mask: objs["genEs"][mask, 1].delta_r(objs["genEs"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genEs"]) > 1,
    ),
    "genAs_toE_lxy_EEdR_XXLowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0,150, name="genAs_lxy"),
                   lambda objs, mask: lxy(objs["genAs_toE"])),
            h.Axis(hist.axis.Regular(25, 0, 0.04, name="genE_genE_dR_lowRange"),
                   lambda objs, mask: objs["genEs"][mask, 1].delta_r(objs["genEs"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genEs"]) > 1,
    ),
    "genAs_toE_lxy_pt_lowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0, 150, name="genAs_lxy"),
                   lambda objs, mask: lxy(objs["genAs_toE"])),
            h.Axis(hist.axis.Regular(25, 0,200, name="genAs_pt"),
                   lambda objs, mask: abs(objs["genAs_toE"].pt)),
        ],
    ),
    "genAs_toE_pt_lxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(140, 0,700, name="genAs_pt"),
                   lambda objs, mask: abs(objs["genAs_toE"].pt)),
            h.Axis(hist.axis.Regular(50, 0, 200, name="genAs_lxy"),
                   lambda objs, mask: lxy(objs["genAs_toE"])),
        ],
    ),
    "genAs_toE_matched_egmLj_pt_lxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(140, 0, 700, name="genAs_pt"),
                   lambda objs, mask: abs(derived_objs["genAs_toE_matched_egmLj"](objs, 0.4).pt)),
            h.Axis(hist.axis.Regular(50, 0, 200, name="genAs_lxy"),
                   lambda objs, mask: lxy(derived_objs["genAs_toE_matched_egmLj"](objs, 0.4))),
        ],
    ),
    "genAs_toMu_pt_lxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0, 500, name="genAs_lxy", label =r"$Z_d$ $L_{xy}$"),
                   lambda objs, mask: lxy(objs["genAs_toMu"])),
            h.Axis(hist.axis.Regular(50, 0,700, name="genAs_pt", label =r"$Z_d$ $p_T$"),
                   lambda objs, mask: abs(objs["genAs_toMu"].pt)),
        ],
    ),
    "genAs_toMu_matched_muLj_pt_lxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(50, 0, 500, name="genAs_lxy"),
                   lambda objs, mask: lxy(derived_objs["genAs_toMu_matched_muLj"](objs, 0.4))),
            h.Axis(hist.axis.Regular(140, 0, 700, name="genAs_pt", label =r"$Z_d$ $p_T$"),
                   lambda objs, mask: abs(derived_objs["genAs_toMu_matched_muLj"](objs, 0.4).pt)),
        ],
    ),
    "genAs_toMu_pt_MuMudR_highRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0,700, name="genAs_pt"),
                   lambda objs, mask: abs(objs["genAs_toMu"].pt)),
            h.Axis(hist.axis.Regular(25, 0, 0.3, name="genMu_genMu_dR_lowRange"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    "genAs_toMu_pt_highRange_MuMudR_lowRange": h.Histogram(
        [
            h.Axis(hist.axis.Regular(25, 0,700, name="genAs_pt"),
                   lambda objs, mask: abs(objs["genAs_toMu"].pt)),
            h.Axis(hist.axis.Regular(25, 0, 0.04, name="genMu_genMu_dR_lowRange"),
                   lambda objs, mask: objs["genMus"][mask, 1].delta_r(objs["genMus"][mask, 0])),
        ],
        evt_mask=lambda objs: ak.num(objs["genMus"]) > 1,
    ),
    # genA-genA
    "genAs_genAs_absdphi": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, math.pi, name="genAs_genAs_absdphi",
                                     label=r"|$\Delta\phi$ between $Z_d$|"),
                   lambda objs, mask: abs(objs["genAs"][mask, 1].delta_phi(objs["genAs"][mask, 0]))),
        ],
        evt_mask=lambda objs: ak.num(objs["genAs"]) > 1,
    ),
    # genA-LJ
    "genAs_lj_dR": h.Histogram(
        [
            # dR(A, nearest LJ)
            h.Axis(hist.axis.Regular(200, 0, 2*math.pi, name="genAs_lj_dR"),
                   lambda objs, mask: dR(objs["genAs"], objs["ljs"]))
        ],
    ),
    "genAs_toE_lj_dR": h.Histogram(
        [
            # dR(A, nearest LJ)
            h.Axis(hist.axis.Regular(200, 0, 2*math.pi, name="genAs_toE_lj_dR"),
                   lambda objs, mask: dR(objs["genAs_toE"], objs["ljs"]))
        ],
    ),
    "genAs_lj_dR_lowRange": h.Histogram(
        [
            # dR(A, nearest LJ)
            h.Axis(hist.axis.Regular(200, 0, 1.0, name="genAs_lj_dR_lowRange"),
                   lambda objs, mask: dR(objs["genAs"], objs["ljs"]))
        ],
    ),
    # genA - LJ 0.4 matching radius, pT Ratios
    "genA_lj_ptRatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(200, 0, 2.0, name="genA_lj_ptRatio",
                   label=r"Lepton Jet pT / (closest) $Z_d$ pT"),
                   lambda objs, mask: objs["ljs"].pt
                       / objs["ljs"].nearest(objs["genAs"], threshold=0.4).pt),
        ],
    ),
    "genA_egmLj_ptRatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(200, 0, 2.0, name="genA_egmLj_ptRatio",
                   label=r"EGM Lepton Jet pT / (closest) $Z_d$ pT"),
                   lambda objs, mask: objs["egm_ljs"].pt
                       / objs["egm_ljs"].nearest(objs["genAs_toE"], threshold=0.4).pt),
        ],
    ),
    "genA_oneElectronLj_ptRatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(200, 0, 2.0, name="genA_oneElectronLj_ptRatio",
                   label=r"(1) Electron Lepton Jet / (closest) $Z_d$ pT"),
                   lambda objs, mask: derived_objs["n_electron_ljs"](objs, 1).pt
                       / derived_objs["n_electron_ljs"](objs, 1).nearest(objs["genAs_toE"], threshold=0.4).pt),
        ],
    ),
    "genA_twoElectronLj_ptRatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(200, 0, 2.0, name="genA_twoElectronLj_ptRatio",
                   label=r"(2) Electron Lepton Jet / (closest) $Z_d$ pT"),
                   lambda objs, mask: derived_objs["n_electron_ljs"](objs, 2).pt
                       / derived_objs["n_electron_ljs"](objs, 2).nearest(objs["genAs_toE"], threshold=0.4).pt),
        ],
    ),
    "genA_onePhotonLj_ptRatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(200, 0, 2.0, name="genA_onePhotonLj_ptRatio",
                   label=r"(1) Photon Lepton Jet / (closest) $Z_d$ pT"),
                   lambda objs, mask: derived_objs["n_photon_ljs"](objs, 1).pt
                       / derived_objs["n_photon_ljs"](objs, 1).nearest(objs["genAs_toE"], threshold=0.4).pt),
        ],
    ),
    "genA_twoPhotonLj_ptRatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(200, 0, 2.0, name="genA_twoPhotonLj_ptRatio",
                   label=r"(2) Photon Lepton Jet / (closest) $Z_d$ pT"),
                   lambda objs, mask: derived_objs["n_photon_ljs"](objs, 2).pt
                       / derived_objs["n_photon_ljs"](objs, 2).nearest(objs["genAs_toE"], threshold=0.4).pt),
        ],
    ),
    "genA_muLj_ptRatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(200, 0, 2.0, name="genA_muLj_ptRatio",
                   label=r"Muon Lepton Jet pT / (closest) $Z_d$ pT"),
                   lambda objs, mask: objs["mu_ljs"].pt
                       / objs["mu_ljs"].nearest(objs["genAs_toMu"], threshold=0.4).pt),
        ],
    ),
    "genA_dsaMuonLj_ptRatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(200, 0, 2.0, name="genA_dsaMuonLj_ptRatio",
                   label=r"DSA Muon Lepton Jet pT / (closest) $Z_d$ pT"),
                   lambda objs, mask: (objs["dsaMuons"][mask]).nearest(objs["ljs"][mask], threshold=0.4).pt
                       / (objs["dsaMuons"][mask]).nearest(objs["genAs_toMu"][mask], threshold=0.4).pt),
        ],
    ),
    "genA_pfMuonLj_ptRatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(200, 0, 2.0, name="genA_pfMuonLj_ptRatio",
                   label=r"PF Muon Lepton Jet pT / (closest) $Z_d$ pT"),
                   lambda objs, mask: (objs["muons"][mask]).nearest(objs["ljs"][mask], threshold=0.4).pt
                       / (objs["muons"][mask]).nearest(objs["genAs_toMu"][mask], threshold=0.4).pt),
        ],
    ),
    "genA_dsaMuon0Lj_ptRatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(200, 0, 2.0, name="genA_dsaMuonLj_ptRatio",
                   label=r"Lead DSA Muon Lepton Jet / (closest) $Z_d$ pT"),
                   lambda objs, mask: ((objs["dsaMuons"][mask, 0:1]).nearest(objs["ljs"][mask], threshold=0.4).pt
                       / (objs["dsaMuons"][mask, 0:1]).nearest(objs["genAs_toMu"][mask], threshold=0.4).pt)),
        ],
        evt_mask=lambda objs: ((ak.num(matched(objs["dsaMuons"], objs["genAs_toMu"], 0.4)) > 0)
                               & (ak.num(matched(objs["dsaMuons"], objs["ljs"], 0.4)) > 0)),
    ),
    "genA_pfMuon0Lj_ptRatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(200, 0, 2.0, name="genA_pfMuonLj_ptRatio",
                   label=r"Lead PF Muon Lepton Jet / (closest) $Z_d$ pT"),
                   lambda objs, mask: ((objs["muons"][mask, 0:1]).nearest(objs["ljs"][mask], threshold=0.4).pt
                       / (objs["muons"][mask, 0:1]).nearest(objs["genAs_toMu"][mask], threshold=0.4).pt)),
        ],
        evt_mask=lambda objs: ((ak.num(matched(objs["muons"], objs["genAs_toMu"], 0.4)) > 0)
                               & (ak.num(matched(objs["muons"], objs["ljs"], 0.4)) > 0)),
    ),
    # genA - LJ 0.4 matching radius, LJ Reco Lxy / True Lxy
    "genA_muLj_lxyRatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(200, 0, 2.0, name="genA_muLj_lxyRatio",
                                    label=r"Muon Lepton Jet Reco L$_{xy}$ / (closest) $Z_d$ L$_{xy}$"),
                   lambda objs, mask: objs["mu_ljs"].kinvtx.lxy
                       / lxy(objs["mu_ljs"].nearest(objs["genAs"], threshold=0.4))),
        ],
    ),
    "genA_egmLj_lxyRatio": h.Histogram(
        [
            h.Axis(hist.axis.Regular(200, 0, 2.0, name="genA_egmLj_lxyRatio",
                                    label=r"EGM Lepton Jet Reco L$_{xy}$ / (closest) $Z_d$ L$_{xy}$"),
                   lambda objs, mask: objs["egm_ljs"].kinvtx.lxy
                       / lxy(objs["egm_ljs"].nearest(objs["genAs"], threshold=0.4))),
        ],
    ),
    # LJ Res vs Reco Lxy
    "mu_lj_genA_ptRatio_vs_recolxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(200, 0, 2.0, name="mu_lj_genA_ptRatio"),
                   lambda objs, mask: objs["mu_ljs"].pt
                       / objs["mu_ljs"].nearest(objs["genAs"]).pt),
            h.Axis(hist.axis.Regular(100, 0, 300, name="mu_lj_recolxy"),
                   lambda objs, mask: objs["mu_ljs"].kinvtx.lxy),
        ],
    ),
    "egm_lj_genA_ptRatio_vs_recolxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(200, 0, 2.0, name="egm_lj_genA_ptRatio"),
                   lambda objs, mask: objs["egm_ljs"].pt
                       / objs["egm_ljs"].nearest(objs["genAs"]).pt),
            h.Axis(hist.axis.Regular(100, 0, 300, name="egm_lj_recolxy"),
                   lambda objs, mask: objs["egm_ljs"].kinvtx.lxy),
        ],
    ),
    # LJ Res vs True Lxy, 0.4 thresholds on dR matching
    "egm_lj_genA_ptRatio_vs_truelxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(200, 0, 2.0, name="egm_lj_genA_ptRatio"),
                   lambda objs, mask: objs["egm_ljs"].pt
                       / objs["egm_ljs"].nearest(objs["genAs"], threshold=0.4).pt),
            h.Axis(hist.axis.Regular(100, 0, 300, name="egm_lj_truelxy"),
                   lambda objs, mask: lxy(objs["egm_ljs"].nearest(objs["genAs"], threshold=0.4))),
        ],
    ),
    "mu_lj_genA_ptRatio_vs_truelxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(200, 0, 2.0, name="mu_lj_genA_ptRatio"),
                   lambda objs, mask: objs["mu_ljs"].pt
                       / objs["mu_ljs"].nearest(objs["genAs"], threshold=0.4).pt),
            h.Axis(hist.axis.Regular(100, 0, 300, name="mu_lj_truelxy"),
                   lambda objs, mask: lxy(objs["mu_ljs"].nearest(objs["genAs"], threshold=0.4))),
        ],
    ),
    "dsaMuon0_genMu0_ptRatio_vs_truelxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0., 2.0, name="dsaMuon0_genMu0_ptRatio"),
                   lambda objs, mask: (objs["dsaMuons"][mask,0:1].pt
                       / objs["dsaMuons"][mask,0:1].nearest(objs["genMus"][mask], threshold=0.4).pt)),
            h.Axis(hist.axis.Regular(100, 0, 300, name="dsaMuon0_lj_truelxy"),
                   lambda objs, mask: lxy(objs["dsaMuons"][mask,0:1].nearest(objs["genAs"][mask], threshold=0.4))),
        ],
        evt_mask=lambda objs: ((ak.num(matched(objs["genMus"], objs["dsaMuons"], 0.4)) > 0)
                               & (ak.num(matched(objs["genAs"], objs["dsaMuons"], 0.4)) > 0)),
    ),
    "muon0_genMu0_ptRatio_vs_truelxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0., 2.0, name="muon0_genMu0_ptRatio"),
                   lambda objs, mask: (objs["muons"][mask,0:1].pt
                       / objs["muons"][mask,0:1].nearest(objs["genMus"][mask], threshold=0.4).pt)),
            h.Axis(hist.axis.Regular(100, 0, 300, name="pfMuon0_lj_truelxy"),
                   lambda objs, mask: lxy(objs["muons"][mask,0:1].nearest(objs["genAs"][mask], threshold=0.4))),
        ],
        evt_mask=lambda objs: ((ak.num(matched(objs["genMus"], objs["muons"], 0.4)) > 0)
                               & (ak.num(matched(objs["genAs"], objs["muons"], 0.4)) > 0)),
    ),
    # LJ Res vs True pT, dR 0.4 matching window
    "dsaMuon0_genMu0_ptRatio_vs_truept": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 2.0, name="dsaMuon0_genMu0_ptRatio"),
                   lambda objs, mask: (objs["dsaMuons"][mask,0:1].pt
                       / objs["dsaMuons"][mask,0:1].nearest(objs["genMus"][mask], threshold=0.4).pt)),
            h.Axis(hist.axis.Regular(200, 0, 1000, name="genMu0_pt"),
                   lambda objs, mask: (objs["dsaMuons"][mask,0:1].nearest(objs["genMus"][mask], threshold=0.4).pt)),
        ],
        evt_mask=lambda objs: (ak.num(objs["dsaMuons"]) > 0),
    ),
    "muon0_genMu0_ptRatio_vs_truept": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 2.0, name="muon0_genMu0_ptRatio"),
                   lambda objs, mask: (objs["muons"][mask,0:1].pt
                       / objs["muons"][mask,0:1].nearest(objs["genMus"][mask], threshold=0.4).pt)),
            h.Axis(hist.axis.Regular(200, 0, 1000, name="genMu0_pt"),
                   lambda objs, mask: (objs["muons"][mask,0:1].nearest(objs["genMus"][mask], threshold=0.4).pt)),
        ],
        evt_mask=lambda objs: (ak.num(objs["muons"]) > 0),
    ),
    "dsaMuon0_muLj_ptRatio_vs_truept": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0, 2.0, name="dsaMuon0_genMu0_ptRatio"),
                   lambda objs, mask: (objs["dsaMuons"][mask,0:1].nearest(objs["ljs"][mask], threshold=0.4).pt
                       / objs["dsaMuons"][mask,0:1].nearest(objs["genAs"][mask], threshold=0.4).pt)),
            h.Axis(hist.axis.Regular(200, 0, 1000, name="genMu0_pt"),
                   lambda objs, mask: (objs["dsaMuons"][mask,0:1].nearest(objs["genMus"][mask], threshold=0.4).pt)),
        ],
        evt_mask=lambda objs: (ak.num(objs["dsaMuons"]) > 0),
    ),
    "muon0_muLj_ptRatio_vs_truept": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0., 2.0, name="muon0_muLj_ptRatio"),
                   lambda objs, mask: (objs["muons"][mask,0:1].nearest(objs["ljs"][mask], threshold=0.4).pt
                       / objs["muons"][mask,0:1].nearest(objs["genAs"][mask], threshold=0.4).pt)),
            h.Axis(hist.axis.Regular(200, 0, 1000, name="genMu0_pt"),
                   lambda objs, mask: (objs["muons"][mask,0:1].nearest(objs["genMus"][mask], threshold=0.4).pt)),
        ],
        evt_mask=lambda objs: ((ak.num(matched(objs["ljs"], objs["muons"], 0.4)) > 0)
                               & (ak.num(matched(objs["genMus"], objs["muons"], 0.4)) > 0)),
    ),
    "egmLj_ptRatio_vs_egm_truept": h.Histogram(
        [
            h.Axis(hist.axis.Regular(100, 0., 2.0, name="egm_lj_genA_ptRatio"),
                   lambda objs, mask: objs["egm_ljs"].pt
                       / objs["egm_ljs"].nearest(objs["genAs"], threshold=0.4).pt),
            h.Axis(hist.axis.Regular(100, 0, 1000, name="genE_pt"),
                   lambda objs, mask: (objs["egm_ljs"].nearest(objs["genEs"], threshold=0.4).pt)[mask,0]),
        ],
    ),
    # LJ True pT vs True Lxy, dR 0.4 matching window
    "genMu0_truept_vs_dsaMuon0_lxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(200, 0, 1000, name="genMu0_pt"),
                   lambda objs, mask: (objs["dsaMuons"][mask,0:1].nearest(objs["genMus"][mask], threshold=0.4).pt)),
            h.Axis(hist.axis.Regular(100, 0, 300, name="dsaMuon0_lj_truelxy"),
                   lambda objs, mask: lxy(objs["dsaMuons"][mask,0:1].nearest(objs["genAs"][mask], threshold=0.4))),
        ],
        evt_mask=lambda objs: ((ak.num(matched(objs["genMus"], objs["dsaMuons"], 0.4)) > 0)
                               & (ak.num(matched(objs["genAs"], objs["dsaMuons"], 0.4)) > 0)),
    ),
    "genMu0_truept_vs_muon0_lxy": h.Histogram(
        [
            h.Axis(hist.axis.Regular(200, 0, 1000, name="genMu0_pt"),
                   lambda objs, mask: (objs["muons"][mask,0:1].nearest(objs["genMus"][mask], threshold=0.4).pt)),
            h.Axis(hist.axis.Regular(100, 0, 300, name="pfMuon0_lj_truelxy"),
                   lambda objs, mask: lxy(objs["muons"][mask,0:1].nearest(objs["genAs"][mask], threshold=0.4))),
        ],
        evt_mask=lambda objs: ((ak.num(matched(objs["genMus"], objs["muons"], 0.4)) > 0)
                               & (ak.num(matched(objs["genAs"], objs["muons"], 0.4)) > 0)),
    ),
}
