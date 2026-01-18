import numpy as np
import pandas as pd
import dataLoaderUtils
import utils
import awkward_pandas
from array import array
import multiprocessing
import tqdm
import sys

def _init_worker(items):
    import utils
    utils._ITEMS = items

def _call_sampleEvent(Enu):
    import utils
    return utils.sampleEvent((Enu, *utils._ITEMS))

if __name__ == "__main__":  # required on macOS / Windows
  nCPU = multiprocessing.cpu_count()-1

  ##################
  ##RUN PARAMETERS##
  ##################
  if len(sys.argv)==2:
    output_name=sys.argv[1]
  else:
    output_name = "test.root"
  if output_name.endswith(".root"):
     outputFormat="root"
  else:
     outputFormat="ascii"
  nps = 1000000
  angle_sampling_type = "NEWTON" #'MuDAR' uses haxton angle plot for muDAR source, 
                                #'NEWTON' uses default NEWTON angle sampling

  #############################
  ##NEUTRINO SOURCE PARAMTERS##
  #############################
  neutrinoSource = "MuDAR" #only 'MuDAR' currently.
  neutrino_direction = [0,0,1]

  ################
  ##DEFINE GRIDS##
  ################
  #Define energy grid
  interpolationStep_MeV = 0.01 #Interpolate digitized xs plots with this precision
  maxEnergy_MeV = 54
  energies_MeV_interp = np.arange(0,maxEnergy_MeV,interpolationStep_MeV)
  #Define angle grid
  interpolationStep_deg = 1.
  maxAngle_deg = 180. 
  angles_deg_interp = np.arange(0,maxAngle_deg,interpolationStep_deg) #Note keeping 180 may cause issues

  ##########################
  ##SPECIFY DATA LOCATIONS##
  ##########################
  #Inclusive xs, CSV of format Energy point [MeV], cross section [cm2]
  #First point should always be 15.4,0.0 for proper interpolations
  total_xs_fname = "data/total_xs/Haxton1987_NEWTON.csv"

  #Exclusive cross sections from the Nakazato paper. Format is
  #Threshold [MeV], J, parity[1=-,0=+], xs@20 MeV [cm2], xs@40 MeV [cm2], xs@60 MeV [cm2]
  partial_xs_fname = "data/partial_xs/Nakazato2018.csv"

  #Lepton angular distribution. Of format
  #Energy [MeV], xs@15 deg. [cm2], Energy [MeV], xs@30 deg., ..., xs@165 deg [cm2]
  MuDAR_lepton_angle_fname = "data/lepton_angle/Haxton1987_muDAR.csv"
  SN_lepton_angle_fname = ""

  newton_lepton_angle_folder = "data/lepton_angle_NEWTON/"
  #Ensure folder name has proper format
  if not newton_lepton_angle_folder.endswith("/"):
      newton_lepton_angle_folder+="/"

  #From NucDeEx
  #Filenames of format: Ex_00.250_ldmodel2_parity1.root
  deexcitation_folder = "data/nucdeex/"
  #Ensure folder name has proper format
  if not deexcitation_folder.endswith("/"):
      deexcitation_folder+="/"

  newton_lepton_angle_folder = "data/lepton_angle_NEWTON/"
  #Ensure folder name has proper format
  if not newton_lepton_angle_folder.endswith("/"):
      newton_lepton_angle_folder+="/"
  #######################
  ##Calculate threshold##
  #######################
  #(160 + 8 electrons) + nu_e -> (16F + 8 electrons) + 1 electron 
  threshold = (utils.atomic_mass_16F_amu-utils.atomic_mass_16O_amu)*utils.amu_to_MeV
  print(f"Threshold of 16O CC process is {threshold:.3f} MeV")
  
  #################
  ##Load total XS##
  #################
  energies_MeV_raw,xs_cm2_raw = dataLoaderUtils.loadCrossSectionCSV(total_xs_fname,threshold==threshold)

  #Log-linear inteerpolation
  with np.errstate(divide='ignore', invalid='ignore'):
      logxs_cm2_raw = np.log10(xs_cm2_raw)
  logxs_interp = np.interp(energies_MeV_interp, energies_MeV_raw, logxs_cm2_raw,left=-np.inf, right=logxs_cm2_raw[-1])
  xs_cm2_interp = np.power(10,logxs_interp)

  #################################
  ##LOAD EXCLUSIVE CROSS SECTIONS##
  #################################
  excitedLevels_MeV,excitedLevels_J,excitedLevels_parity,xs20,xs40,xs60 = dataLoaderUtils.loadNazakatoPartialXSData(partial_xs_fname,xs_threshold=1e-46)

  ###############################
  ##SOLVE NAKAZATO COEFFICIENTS##
  ###############################
  c1s,c2s,c3s = utils.solveNakazatoCoeffs(excitedLevels_MeV,xs20,xs40,xs60)

  #########################################################
  ##CALCULATE PARTIAL CROSS SECTIONS OVER OUR ENERGY GRID##
  #########################################################
  partial_xs = utils.calcNakazatoPartialXS(excitedLevels_MeV,energies_MeV_interp,c1s,c2s,c3s)

  #################################################
  ##NORMALIZE PARTIAL CROSS SECTIONS FOR SAMPLING##
  #################################################
  Z = np.vstack([np.asarray(xs) for xs in partial_xs])
  #Convert to probabilities
  colsum = Z.sum(axis=0)                                
  excitation_probs = np.zeros_like(Z, dtype=float)
  nonzero_mask = colsum > 0
  excitation_probs[:, nonzero_mask] = Z[:, nonzero_mask] / colsum[nonzero_mask]

  ###############################
  ##CALCULATE NEUTRINO SPECTRUM##
  ###############################
  dar_spectrum = 96*np.power(energies_MeV_interp,2) * np.power(utils.mass_mu_MeV,-4)  * (utils.mass_mu_MeV - 2*energies_MeV_interp)
  dar_spectrum = np.where(dar_spectrum > 0, dar_spectrum, 0) #Function turns negative at high energies, replace with zeros

  #TODO: Supernova spectrum

  if neutrinoSource=="MuDAR":
    spectrum = dar_spectrum
  else:
    spectrum = ""

  ##########################################################
  ##CALCULATE FOLDED SPECTRUM, FLUX-AVERAGED CROSS SECTION##
  ##########################################################
  folded_spectrum = np.multiply(xs_cm2_interp,spectrum)
  neutrino_energy_probs = folded_spectrum / folded_spectrum.sum()

  #Calculate flux-averaged spectrum
  num = np.trapezoid(xs_cm2_interp * dar_spectrum, energies_MeV_interp)
  den = np.trapezoid(dar_spectrum, energies_MeV_interp)
  flux_averaged_xs = num / den
  flux_averaged_xs_naturalunits = flux_averaged_xs/utils.hbar_c_squared
  print(f"Flux-averaged cross section is {flux_averaged_xs:.3e} cm2")

  ################################################
  ##LOAD LEPTON ANGLE DISTRIBUTIONS, INTERPOLATE##
  ################################################
  if angle_sampling_type=="MuDAR":
    lepton_angles_raw,lepton_energies_raw,lepton_angle_xs_raw = dataLoaderUtils.loadHaxtonMuDARAngles(MuDAR_lepton_angle_fname)
    #Intepolate the above plot over our energy/angle grid
    angle_vs_lepton_energy_vs_xs_cm2 = utils.interpolateHaxtonMuDARAngles(lepton_angles_raw,angles_deg_interp,lepton_energies_raw,energies_MeV_interp,lepton_angle_xs_raw)

    #Modify angular distribution based on solid angle subtended by each theta
    theta_rad_interp = np.deg2rad(angles_deg_interp)
    sin_weights = np.sin(theta_rad_interp)
    weighted_xs = angle_vs_lepton_energy_vs_xs_cm2 * sin_weights[:, np.newaxis]

    #Create probabilitiy distributions, normalized per lepton energy
    lepton_angle_probs = np.zeros_like(weighted_xs)
    colsum = weighted_xs.sum(axis=0)
    mask = colsum > 0
    lepton_angle_probs[:, mask] = weighted_xs[:, mask] / colsum[mask]
  else:
    angles_deg_raw, Enus_NEWTON_raw, Enus_vs_angle_raw = dataLoaderUtils.loadNewtonDoubleDiffData(newton_lepton_angle_folder)
    lepton_angle_probs = []
    Enus_vs_lepton_angles_vs_xs_cm2 = []
    for iEx,Enu_energy_dist_raw in enumerate(Enus_NEWTON_raw):
      pdf,unnormalized_pdf = utils.interpolateNEWTONAnglesAndNormalize(
                                                                angles_deg_raw,
                                                                angles_deg_interp,
                                                                Enus_NEWTON_raw[iEx],
                                                                energies_MeV_interp,
                                                                Enus_vs_angle_raw[iEx]
                                                                )
      lepton_angle_probs.append(pdf)
    lepton_angle_probs = np.array(lepton_angle_probs)

  ###########################
  ##Load de-excitation data##
  ###########################
  ex_dfs = dataLoaderUtils.loadNucDeExData(deexcitation_folder)
    
  #########################################################
  ##Sample neutrino eneries from folded xs times spectrum##
  #########################################################
  sampled_neutrino_energies_MeV = np.random.choice(energies_MeV_interp, size=nps, p=neutrino_energy_probs)

  #For validation only
  sampled_lepton_energies = []
  calculated_lepton_energies = []
  sampled_thetas_deg = []

  #Wrap up args to even sampler
  items = [energies_MeV_interp,excitedLevels_MeV,excitedLevels_J,excitedLevels_parity,excitation_probs,ex_dfs,
          angles_deg_interp,lepton_angle_probs,
          neutrino_direction,utils.nuc_mass_16O_MeV,utils.nuc_mass_16F_MeV,
          "16O","16F","ve","electron",angle_sampling_type]
  
  with multiprocessing.Pool(nCPU,initializer=_init_worker,initargs=(items,)) as pool:
    results = list(tqdm.tqdm(pool.imap(_call_sampleEvent, sampled_neutrino_energies_MeV, chunksize=200),total=len(sampled_neutrino_energies_MeV)))
    
  #########
  ##Write##
  #########
  if outputFormat=="ascii":
    outFile = open(output_name,"w")
    #Generate our events in MARLEY format
    line=str(flux_averaged_xs_naturalunits)
    outFile.write(line)

    for res in results:
      header,inParticles,outParticles = res[i],res[i],res[i]
      
      #Event header
      Ni = len(inParticles)
      Nf = len(outParticles)
      Ex = header["Ex"]
      twoJ = header["twoJ"]
      parity = header["parity"]
      line = f"\n{Ni} {Nf} {Ex:.17e} {twoJ} {parity}"
      outFile.write(line)

      for ipart,part in enumerate(inParticles):
        #Initial neutrino
        line = f'\n{part["pdg"]} {part["totalE"]:.17e} {part["PX"]:.17e} {part["PY"]:.17e} {part["PZ"]:.17e} {part["mass"]:.17e} {part["charge"]}'
        outFile.write(line)

      for ipart,part in enumerate(outParticles):
        #Initial neutrino
        line = f'\n{part["pdg"]} {part["totalE"]:.17e} {part["PX"]:.17e} {part["PY"]:.17e} {part["PZ"]:.17e} {part["mass"]:.17e} {part["charge"]}'
        outFile.write(line)
    #Write ascii output 
    outFile.close()

  elif outputFormat=="root":
    import ROOT
    # Write ROOT with std::vector branches (no n... counter branches)
    f = ROOT.TFile(output_name, "RECREATE")

    # ----------------
    # headerTree
    # ----------------
    headerTree = ROOT.TTree("headerTree", "")
    fluxAveraged_xs = array("d", [float(flux_averaged_xs)])  # double
    headerTree.Branch("fluxAveraged_xs", fluxAveraged_xs, "fluxAveraged_xs/D")
    headerTree.Fill()
    headerTree.Write()

    # ----------------
    # eventTree
    # ----------------
    t = ROOT.TTree("eventTree", "")

    eventNum  = array("i", [0])    # int32
    Enu       = array("f", [0.0])  # float32
    Ex        = array("f", [0.0])  # float32
    twoJ      = array("h", [0])    # int16
    parity    = array("b", [0])    # int8  (+1/-1)
    theta_deg = array("f", [0.0])  # float32

    t.Branch("eventNum", eventNum, "eventNum/I")
    t.Branch("Enu", Enu, "Enu/F")
    t.Branch("Ex", Ex, "Ex/F")
    t.Branch("twoJ", twoJ, "twoJ/S")
    t.Branch("parity", parity, "parity/B")
    t.Branch("theta_deg", theta_deg, "theta_deg/F")

    inPDG    = ROOT.std.vector("int")()
    inMass   = ROOT.std.vector("float")()
    inKE     = ROOT.std.vector("float")()
    inPx     = ROOT.std.vector("float")()
    inPy     = ROOT.std.vector("float")()
    inPz     = ROOT.std.vector("float")()
    inCharge = ROOT.std.vector("float")()

    outPDG    = ROOT.std.vector("int")()
    outMass   = ROOT.std.vector("float")()
    outKE     = ROOT.std.vector("float")()
    outPx     = ROOT.std.vector("float")()
    outPy     = ROOT.std.vector("float")()
    outPz     = ROOT.std.vector("float")()
    outCharge = ROOT.std.vector("float")()

    t.Branch("inParticlePDG", inPDG)
    t.Branch("inParticleMass", inMass)
    t.Branch("inParticleKE", inKE)
    t.Branch("inParticlePx", inPx)
    t.Branch("inParticlePy", inPy)
    t.Branch("inParticlePz", inPz)
    t.Branch("inParticleCharge", inCharge)

    t.Branch("outParticlePDG", outPDG)
    t.Branch("outParticleMass", outMass)
    t.Branch("outParticleKE", outKE)
    t.Branch("outParticlePx", outPx)
    t.Branch("outParticlePy", outPy)
    t.Branch("outParticlePz", outPz)
    t.Branch("outParticleCharge", outCharge)

    n_events = len(results)
    for i,res in enumerate(results):
      headerList, inParticleList, outParticleList = res[0], res[1], res[2]
      eventNum[0]  = i
      Enu[0]       = float(headerList["Enu"])
      Ex[0]        = float(headerList["Ex"])
      twoJ[0]      = int(headerList["twoJ"])
      parity[0]    = 1 if headerList["parity"] == "+" else -1
      theta_deg[0] = float(headerList["theta_deg"])

      inPDG.clear(); inMass.clear(); inKE.clear(); inPx.clear(); inPy.clear(); inPz.clear(); inCharge.clear()
      for p in inParticleList:
        inPDG.push_back(int(p["pdg"]))
        inMass.push_back(float(p["mass"]))
        inKE.push_back(float(p["totalE"] - p["mass"]))
        inPx.push_back(float(p["PX"]))
        inPy.push_back(float(p["PY"]))
        inPz.push_back(float(p["PZ"]))
        inCharge.push_back(float(p["charge"]))

      outPDG.clear(); outMass.clear(); outKE.clear(); outPx.clear(); outPy.clear(); outPz.clear(); outCharge.clear()
      for p in outParticleList:
        outPDG.push_back(int(p["pdg"]))
        outMass.push_back(float(p["mass"]))
        outKE.push_back(float(p["totalE"] - p["mass"]))
        outPx.push_back(float(p["PX"]))
        outPy.push_back(float(p["PY"]))
        outPz.push_back(float(p["PZ"]))
        outCharge.push_back(float(p["charge"]))

      t.Fill()
  t.Write()
  f.Close()
