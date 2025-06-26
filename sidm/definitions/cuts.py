"""Define all available cuts"""

# columnar analysis
import awkward as ak
# local
from sidm.definitions.objects import derived_objs
from sidm.tools.utilities import dR, lxy, rho, check_bits, returnBitMapTArrayPhoton

obj_cut_defs = {
    "pvs": {
        "ndof > 4": lambda objs: objs["pvs"].ndof > 4,
        "|z| < 24 cm": lambda objs: abs(objs["pvs"].z) < 24,
        "|rho| < 0.02 cm": lambda objs: rho(objs["pvs"], ref=objs["bs"]) < 0.02,
    },
    "ljs": {
        "pT > 10 GeV": lambda objs: objs["ljs"].pt > 10,
        "pT > 20 GeV": lambda objs: objs["ljs"].pt > 20,
        "pT > 30 GeV": lambda objs: objs["ljs"].pt > 30,
        "pT > 40 GeV": lambda objs: objs["ljs"].pt > 40,
        "pT > 50 GeV": lambda objs: objs["ljs"].pt > 50,
        "pT > 60 GeV": lambda objs: objs["ljs"].pt > 60,
        "pT < 150 GeV": lambda objs: objs["ljs"].pt < 150,
        "|eta| < 2.4": lambda objs: abs(objs["ljs"].eta) < 2.4,
        "mu_charge == 0": lambda objs: ak.sum(objs["ljs"].muons.charge, axis= -1) == 0,
        "dR(LJ, A) < 0.2": lambda objs: dR(objs["ljs"], objs["genAs"]) < 0.2,
        "egmLj": lambda objs: ak.num(objs["ljs"].muons) == 0,
        "eLj": lambda objs: (ak.num(objs["ljs"].muons) == 0) & (ak.num(objs["ljs"].electrons) > 0) & (ak.num(objs["ljs"].photons) == 0),
        "gLj": lambda objs: (ak.num(objs["ljs"].muons) == 0) & (ak.num(objs["ljs"].electrons) == 0) & (ak.num(objs["ljs"].photons) > 0),
        "eAndGLj": lambda objs: (ak.num(objs["ljs"].muons) == 0) & (ak.num(objs["ljs"].electrons) > 0) & (ak.num(objs["ljs"].photons) > 0),
        "muLj": lambda objs: ak.num(objs["ljs"].muons) > 0,
        "pfMuLj": lambda objs: (ak.num(objs["ljs"].pfMuons) > 0) & (ak.num(objs["ljs"].dsaMuons) == 0),
        "dsaMuLj": lambda objs: ak.num(objs["ljs"].dsaMuons) > 0,
        "2dsaMuLj": lambda objs: ak.num(objs["ljs"].dsaMuons) > 2,
        "pfDsaMuLj": lambda objs: (ak.num(objs["ljs"].pfMuons) > 0) & (ak.num(objs["ljs"].dsaMuons) > 0),
    },
    "egm_ljs": {
        "eLj": lambda objs: (objs["egm_ljs"].electron_n > 0) & (objs["egm_ljs"].photon_n == 0),
        "gLj": lambda objs: (objs["egm_ljs"].electron_n == 0) & (objs["egm_ljs"].photon_n > 0),
        "egLj": lambda objs: (objs["egm_ljs"].electron_n > 0) & (objs["egm_ljs"].photon_n > 0),
        "1eLj": lambda objs: (objs["egm_ljs"].electron_n == 1) & (objs["egm_ljs"].photon_n == 0),
        "2eLj": lambda objs: (objs["egm_ljs"].electron_n == 2) & (objs["egm_ljs"].photon_n == 0),
        "1gLj": lambda objs: (objs["egm_ljs"].electron_n == 0) & (objs["egm_ljs"].photon_n == 1),
        "2gLj": lambda objs: (objs["egm_ljs"].electron_n == 0) & (objs["egm_ljs"].photon_n == 2),
    },
    "mu_ljs": {
        "pfMuLj": lambda objs: (objs["mu_ljs"].pfMu_n > 0) & (objs["mu_ljs"].dsaMu_n == 0),
        "dsaMuLj": lambda objs: (objs["mu_ljs"].pfMu_n == 0) & (objs["mu_ljs"].dsaMu_n > 0),
        "pf_dsa_muLj": lambda objs: (objs["mu_ljs"].pfMu_n > 0) & (objs["mu_ljs"].dsaMu_n > 0),
    },
    "genMus":{
        "pT >= 10 GeV": lambda objs: objs["genMus"].pt >= 10,
        "status 1": lambda objs: objs["genMus"].status == 1,
        "fromGenA": lambda objs: abs(objs["genMus"].distinctParent.pdgId) == 32,
        "parent == A": lambda objs: abs(objs["genMus"].parent.pdgId) == 32,
    },
    "genEs":{
        "status 1": lambda objs: objs["genEs"].status == 1,
        "fromGenA": lambda objs: abs(objs["genEs"].distinctParent.pdgId) == 32,
        "parent == A": lambda objs: abs(objs["genEs"].parent.pdgId) == 32,
    },
    "genAs": {
        "dR(A, LJ) < 0.2": lambda objs: dR(objs["genAs"], objs["ljs"]) < 0.2,
        "dR(A, LJ) < 0.4": lambda objs: dR(objs["genAs"], objs["ljs"]) < 0.4,
        "lxy < 10 cm": lambda objs: lxy(objs["genAs"]) < 10,
        "lxy < 40 cm": lambda objs: lxy(objs["genAs"]) < 40,
        "10 cm <= lxy < 100 cm": lambda objs: ((lxy(objs["genAs"]) >= 10)
                                               & (lxy(objs["genAs"]) < 100)),
        "100 cm <= lxy < 135 cm": lambda objs: ((lxy(objs["genAs"]) >= 100)
                                               & (lxy(objs["genAs"]) < 135)),
        "lxy >= 100 cm": lambda objs: lxy(objs["genAs"]) >= 100,
        "lxy <= 100 cm": lambda objs: lxy(objs["genAs"]) <= 100,
        "lxy <= 150 cm": lambda objs: lxy(objs["genAs"]) <= 150,
        "lxy <= 250 cm": lambda objs: lxy(objs["genAs"]) <= 250,
        "lxy <= 400 cm": lambda objs: lxy(objs["genAs_toMu"]) <= 400,
        "pT > 30 GeV": lambda objs: objs["genAs"].pt > 30,
        "pT < 300 GeV": lambda objs: objs["genAs"].pt < 300,
    },
    "genAs_toMu": {
        "dR(A, LJ) < 0.2": lambda objs: dR(objs["genAs_toMu"], objs["ljs"]) < 0.2,
        "dR(A, LJ) < 0.4": lambda objs: dR(objs["genAs_toMu"], objs["ljs"]) < 0.4,
        "lxy < 10 cm": lambda objs: lxy(objs["genAs_toMu"]) < 10,
        "10 cm <= lxy < 100 cm": lambda objs: (lxy(objs["genAs_toMu"]) >= 10) & (lxy(objs["genAs_toMu"]) < 100),
        "lxy <= 60 cm": lambda objs: lxy(objs["genAs_toMu"]) <= 60,
        "lxy >= 100 cm": lambda objs: lxy(objs["genAs_toMu"]) >= 100,
        "lxy <= 100 cm": lambda objs: lxy(objs["genAs_toMu"]) <= 100,
        "lxy <= 150 cm": lambda objs: lxy(objs["genAs_toMu"]) <= 150,
        "lxy <= 250 cm": lambda objs: lxy(objs["genAs_toMu"]) <= 250,
        "lxy <= 400 cm": lambda objs: lxy(objs["genAs_toMu"]) <= 400,
        "pT > 30 GeV": lambda objs: objs["genAs_toMu"].pt > 30,
        "pT < 300 GeV": lambda objs: objs["genAs_toMu"].pt < 300,
    },
    "genAs_toE": {
        "dR(A, LJ) < 0.2": lambda objs: dR(objs["genAs_toE"], objs["ljs"]) < 0.2,
        "dR(A, LJ) < 0.4": lambda objs: dR(objs["genAs_toE"], objs["ljs"]) < 0.4,
        "lxy <= 5 cm": lambda objs: lxy(objs["genAs_toE"]) <= 5,
        "lxy <= 2.5 cm": lambda objs: lxy(objs["genAs_toE"]) <= 2.5,
        "lxy < 10 cm": lambda objs: lxy(objs["genAs_toE"]) < 10,
        "lxy < 40 cm": lambda objs: lxy(objs["genAs_toE"]) < 40,
        "10 cm <= lxy < 100 cm": lambda objs: (lxy(objs["genAs_toE"]) >= 10) & (lxy(objs["genAs_toE"]) < 100),
        "40 cm <= lxy < 77 cm": lambda objs: (lxy(objs["genAs_toE"]) >= 40) & (lxy(objs["genAs_toE"]) < 77),
        "100 cm <= lxy < 135 cm": lambda objs: (lxy(objs["genAs_toE"]) >= 100) & (lxy(objs["genAs_toE"]) < 135),
        "125 cm <= lxy <= 135 cm": lambda objs: (lxy(objs["genAs_toE"]) >= 125) & (lxy(objs["genAs_toE"]) < 135),
        "lxy >= 100 cm": lambda objs: lxy(objs["genAs_toE"]) >= 100,
        "lxy <= 100 cm": lambda objs: lxy(objs["genAs_toE"]) <= 100,
        "lxy <= 150 cm": lambda objs: lxy(objs["genAs_toE"]) <= 150,
        "lxy <= 250 cm": lambda objs: lxy(objs["genAs_toE"]) <= 250,
        "lxy <= 400 cm": lambda objs: lxy(objs["genAs_toMu"]) <= 400,
        "pT > 30 GeV": lambda objs: objs["genAs_toE"].pt > 30,
        "pT < 300 GeV": lambda objs: objs["genAs_toE"].pt < 300,
    },
    "electrons": {
        "pT > 10 GeV": lambda objs: objs["electrons"].pt > 10,
        "pT > 30 GeV": lambda objs: objs["electrons"].pt > 30,
        "|eta| < 1.479": lambda objs: abs(objs["electrons"].eta) < 1.479,
        "1.479 < |eta| < 2.4": lambda objs: ((abs(objs["electrons"].eta) > 1.479)
                                             & (abs(objs["electrons"].eta) < 2.4)),
        "|eta| < 2.4": lambda objs: abs(objs["electrons"].eta) < 2.4,
        "dR(e, A) < 0.5": lambda objs: dR(objs["electrons"], objs["genAs_toE"]) < 0.5,
        "looseID": lambda objs: objs["electrons"].cutBased >= 2,
        "photonIdx": lambda objs: objs["electrons"].photonIdx != -1,
        "barrel": lambda objs: (objs["electrons"].eta + objs["electrons"].deltaEtaSC) <= 1.479 ,
        "endcap": lambda objs: (objs["electrons"].eta + objs["electrons"].deltaEtaSC) > 1.479 ,
        "barrel_sieie": lambda objs: objs["electrons"].sieie <= 0.0112,
        "barrel_iso": lambda objs: objs["electrons"].pfRelIso03_all < (.112 + .506/objs["electrons"].pt),
        "barrel_hoe": lambda objs: objs["electrons"].hoe < 0.05 + 1.16/((1 + objs["electrons"].scEtOverPt) * objs["electrons"].pt) + 0.0324*objs["rho_PFIso"]/((1 + objs["electrons"].scEtOverPt) * objs["electrons"].pt),
        "barrel_eInvMinusPInv": lambda objs: objs["electrons"].eInvMinusPInv < 0.193,
        "lostHits": lambda objs: objs["electrons"].lostHits <= 1,
        "convVeto": lambda objs: objs["electrons"].convVeto == True,
        "endcap_sieie": lambda objs: objs["electrons"].sieie <= 0.0425,
        "endcap_iso": lambda objs: objs["electrons"].pfRelIso03_all < (.108 + .963/objs["electrons"].pt),
        "endcap_hoe": lambda objs: objs["electrons"].hoe <  0.0441 + 2.54/((1 + objs["electrons"].scEtOverPt) * objs["electrons"].pt) + 0.183*objs["rho_PFIso"]/((1 + objs["electrons"].scEtOverPt) * objs["electrons"].pt),
        "endcap_eInvMinusPInv": lambda objs: objs["electrons"].eInvMinusPInv < 0.111,
        'MVANonIsoWPL': lambda objs: objs['electrons'].mvaFall17V2noIso_WPL,
        "missing_hits == 0" : lambda objs: objs["electrons"].lostHits == 0,
        "missing_hits == 1" : lambda objs: objs["electrons"].lostHits == 1,
    },
    "muons": {
        "looseID": lambda objs: objs["muons"].looseId,
        "pT > 5 GeV": lambda objs: objs["muons"].pt > 5,
        "|eta| < 2.4": lambda objs: abs(objs["muons"].eta) < 2.4,
        "dR(mu, A) < 0.5": lambda objs: dR(objs["muons"], objs["genAs_toMu"]) < 0.5,
    },
    "photons":{
        "pT > 20 GeV": lambda objs: objs["photons"].pt > 20,
        "pT > 30 GeV": lambda objs: objs["photons"].pt > 30,
        "|eta| < 2.5": lambda objs: abs(objs["photons"].eta) < 2.5, # fixme: do we want eta or scEta
        "eta": lambda objs: objs["photons"].isScEtaEB | objs["photons"].isScEtaEE,
        "barrel": lambda objs: objs["photons"].isScEtaEB,
        "endcap": lambda objs: objs["photons"].isScEtaEE,
        "looseID": lambda objs: objs["photons"].cutBased >= 1,
        "pixelSeed": lambda objs: objs["photons"].pixelSeed == False ,
        "electronVeto": lambda objs: objs["photons"].electronVeto,
        "dR(gm, A) < 0.5": lambda objs: dR(objs["photons"], objs["genAs_toE"]) < 0.5,
        "barrel_hoe": lambda objs: objs["photons"].hoe <= .04596,
        "barrel_sieie": lambda objs: objs["photons"].sieie <=  0.0106 ,
        "|eta| < 1.479": lambda objs: abs(objs["photons"].eta) < 1.479,
        'Custom Cutbased': lambda objs: returnBitMapTArrayPhoton(objs['photons'].vidNestedWPBitmap, 'PhoFull5x5SigmaIEtaIEtaCut', 'PhoIsoWithEALinScalingCut'),
        'Photon DR Veto 0p025': lambda objs: (dR(objs['photons'],objs['electrons']) > 0.025),
        'Photon DR Veto 0p035': lambda objs: (dR(objs['photons'],objs['electrons']) > 0.035),
        'Photon DR Veto 0p045': lambda objs: (dR(objs['photons'],objs['electrons']) > 0.045),
    },
    "dsaMuons": {
        "pT > 10 GeV": lambda objs: objs["dsaMuons"].pt > 10,
        "|eta| < 2.4": lambda objs: abs(objs["dsaMuons"].eta) < 2.4,
        # displaced ID as a single flag and as individual cuts
        "displaced ID" : lambda objs: objs["dsaMuons"].displacedID > 0,
        "DT + CSC hits > 12": lambda objs: (objs["dsaMuons"].trkNumDTHits
                                            + objs["dsaMuons"].trkNumCSCHits) > 12,
        "ifcsczero": lambda objs: ak.where(((objs["dsaMuons"].trkNumCSCHits == 0)
                                           & (objs["dsaMuons"].trkNumDTHits <= 18)), False, True),
        "normChi2 < 2.5": lambda objs: objs["dsaMuons"].normChi2 < 2.5,
        "ptErrorOverPT < 1": lambda objs: (objs["dsaMuons"].ptErr / objs["dsaMuons"].pt) < 1.0,
        # just use segment-based matching
        "no PF match" : lambda objs: objs["dsaMuons"].muonMatch1/objs["dsaMuons"].nSegments < 0.667,
        "dR(mu, A) < 0.5": lambda objs: dR(objs["dsaMuons"], objs["genAs_toMu"]) < 0.5,
        "dR(dsa, pf) > 0.2": lambda objs: dR(objs["dsaMuons"], objs["muons"]) > 0.2,
        "dR(dsa, pf) > 0.02": lambda objs: dR(objs["dsaMuons"], objs["muons"]) > 0.02,
    },
}

evt_cut_defs = {
    # This following will be True for every event. There's probably a more intuitive way to do this
    "Keep all evts": lambda objs: objs["pvs"].npvs >= 0,
    "pass triggers": lambda objs: (
          objs["hlt"].DoubleL2Mu23NoVtx_2Cha
        | objs["hlt"].DoubleL2Mu23NoVtx_2Cha_NoL2Matched
        | objs["hlt"].DoubleL2Mu23NoVtx_2Cha_CosmicSeed
        | objs["hlt"].DoubleL2Mu23NoVtx_2Cha_CosmicSeed_NoL2Matched
        | objs["hlt"].DoubleL2Mu25NoVtx_2Cha_Eta2p4
        | objs["hlt"].DoubleL2Mu25NoVtx_2Cha_CosmicSeed_Eta2p4
    ),
    ">=1 muon": lambda objs: ak.num(objs["muons"]) >= 1,
    "PV filter": lambda objs: ak.flatten(objs["pvs"].npvsGood) >= 1,
    #"Cosmic veto": lambda objs: objs["cosmicveto"].result,
    ">=2 LJs": lambda objs: ak.num(objs["ljs"]) >= 2,
    ">=2 matched As": lambda objs: ak.num(derived_objs["genAs_matched_lj"](objs, 0.2)) >= 2,
    # 4mu: leading two LJs are both mu-type
    "4mu": lambda objs: ak.count_nonzero(objs["ljs"][:, :2].muon_n >= 2, axis=-1) == 2,
    # 2mu2e: leading two LJs contain exactly 1 mu-type and exactly 1 egm-type
    "2mu2e": lambda objs: ((ak.count_nonzero(objs["ljs"][:, :2].muon_n >= 2, axis=-1) == 1)
                           & (ak.count_nonzero(objs["ljs"][:, :2].muon_n == 0, axis=-1) == 1)),
    "genAs_toE_matched_egmLj": lambda objs: ak.num(derived_objs["genAs_toE_matched_egmLj"](objs, 0.4)) >= 1,
    "genAs_toMu_matched_muLj": lambda objs: ak.num(derived_objs["genAs_toMu_matched_muLj"](objs, 0.4)) >= 1,
    "genAs_toE": lambda objs: ak.num(objs["genAs_toE"]) >= 1,
    "genAs_toMu": lambda objs: ak.num(objs["genAs_toMu"]) >= 1,
    "ljs": lambda objs: ak.num(objs["ljs"]) >= 1,
    "50 GeV <= GenMu0_pT <= 60 GeV": lambda objs : (objs["genMus"][:, 0].pt >=50) & (objs["genMus"][:, 0].pt <=60),
    "genMus": lambda objs: ak.num(objs["genMus"]) > 1,
    "dR(Mu_0, Mu_1) > 0.03": lambda objs: objs["genMus"][:,0].delta_r(objs["genMus"][:,1]) > 0.03,
}
