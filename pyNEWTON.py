#Simple nu-O event generator for 10s-of-MeV neutrinos
#Follows approach taken by NEWTON
#
#Steps:
# Supply total xs (different choices), interpolate
# Supply partial xs, fit coefficients, interpolate
# Specify neutrino flux, fold with XS, sample
# Choose excitation
# Estimate lepton energy, sample direction
# Calculate lepton energy direction, nuclear recoil
# Draw from pre-computed TALYS de-excitation files to generate events
# Write output in ascii format
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
import utils
import sys
import tqdm
import multiprocessing

if __name__ == "__main__":  # required on macOS / Windows
    nCPU = multiprocessing.cpu_count()-1

    #################
    ##CONFIGURATION##
    #################
    output_name = "test.o"
    outputFormat = "ascii" #ascii, MARLEY-style output
    nps = 10000
    neutrino_direction = [0,0,1]

    #Interpolate energies over this grid
    interpolationStep_MeV = 0.01 #Interpolate digitized xs plots with this precision
    maxEnergy_MeV = 53
    energies_MeV_interp = np.arange(0,maxEnergy_MeV,interpolationStep_MeV)
    #Interpolate angles over this grid
    angles_deg_interp = np.arange(0,180,1)

    #Total xs, CSV of format Energy point [MeV], cross section [cm2]
    #First point should always be 15.4,0.0 for proper interpolations
    total_xs_fname = "data/total_xs/Haxton1987_NEWTON.csv"

    #Partial cross sections from the nakazato paper. Format is
    #Threshold [MeV],J, [1=-,0=+], xs@20 MeV [cm2], xs@40 MeV [cm2], xs@60 MeV [cm2]
    partial_xs_fname = "data/partial_xs/Nakazato2018.csv"

    #Lepton angular distribution. Of format
    #Energy [MeV], xs@15 deg. [cm2], Energy [MeV], xs@30 deg., ..., xs@165 deg [cm2]
    muDAR_lepton_angle_fname = "data/lepton_angle/Haxton1987_muDAR.csv"
    SN_lepton_angle_fname = ""

    #From NucDeEx
    #Filenames of format: Ex_00.250_ldmodel2_parity1.root
    #containing "tree" with branches Ex_MeV, size, PDG, mass_MeV, totalE_MeV, KE_MeV, PMag_MeV, PX_MeV, PY_MeV, PZ_MeV 
    deexcitation_folder = "data/nucdeex/"
    if not deexcitation_folder.endswith("/"):
        deexcitation_folder+="/"

    #################
    ##LOAD TOTAL XS##
    #################
    energies_MeV_raw,xs_cm2_raw = utils.loadCrossSectionCSV(total_xs_fname)

    #Log-linear inteerpolation
    with np.errstate(divide='ignore', invalid='ignore'):
        logxs_cm2_raw = np.log10(xs_cm2_raw) #Suppress errors, xs with values equal to zero replaced with -np.inf
    logxs_interp = np.interp(energies_MeV_interp, energies_MeV_raw, logxs_cm2_raw,left=-np.inf,right=logxs_cm2_raw[-1])
    xs_cm2_interp = np.power(10,logxs_interp)

    ###################
    ##LOAD PARTIAL XS##
    ###################
    excitedLevels_MeV,excitedLevels_J,excitedLevels_parity,excitedXS_20MeV,excitedXS_40MeV,excitedXS_60MeV = utils.loadNazakatoPartialXSData(partial_xs_fname)

    ###########################
    ##WRITE PARTIAL XS AS FNS##
    ###########################
    c1s,c2s,c3s = utils.solveNakazatoCoeffs(excitedLevels_MeV,excitedXS_20MeV,excitedXS_40MeV,excitedXS_60MeV)#Calculate partial cross sections on our grid
    partial_xs = utils.calcNakazatoPartialXS(excitedLevels_MeV,energies_MeV_interp,c1s,c2s,c3s)

    Z = np.vstack([np.asarray(xs) for xs in partial_xs])
    #Convert to probabilities
    colsum = Z.sum(axis=0)                                
    excitation_probs = np.zeros_like(Z, dtype=float)
    nonzero_mask = colsum > 0
    excitation_probs[:, nonzero_mask] = Z[:, nonzero_mask] / colsum[nonzero_mask]

    ###########################################
    ##CALC NEUTRINO SPECTRUM, FOLDED SPECTRUM##
    ###########################################
    #SNS neutrino spectrum
    dar_spectrum = 96*np.power(energies_MeV_interp,2) * np.power(utils.mass_mu_MeV,-4)  * (utils.mass_mu_MeV - 2*energies_MeV_interp)
    dar_spectrum = np.where(dar_spectrum > 0, dar_spectrum, 0) #Function turns negative at high energies, replace with zeros

    #TODO: Supernova spectrum

    #Calculate folded cross section
    folded_spectrum = np.multiply(xs_cm2_interp,dar_spectrum)
    #Calculate flux-averaged spectrum
    flux_averaged_xs = np.average(folded_spectrum)
    flux_averaged_xs_naturalunits = flux_averaged_xs/utils.hbar_c_squared
    # Normalize
    neutrino_energy_probs = folded_spectrum / folded_spectrum.sum()

    #########################
    ##LOAD UP LEPTON ANGLES##
    #########################
    #Load up lepton angular distributions
    angles_raw,lepton_energies_raw,haxton_xs_raw = utils.loadHaxtonAngles(muDAR_lepton_angle_fname)

    #Intepolate the above plot over our energy/angle grid
    angle_energy_xs_cm2 = utils.interpolateHaxtonAngles(angles_raw,angles_deg_interp,lepton_energies_raw,energies_MeV_interp,haxton_xs_raw)

    theta_rad_interp = np.deg2rad(angles_deg_interp)
    sin_weights = np.sin(theta_rad_interp)
    weighted_xs = angle_energy_xs_cm2 * sin_weights[:, np.newaxis]

    #Create probabilitiy distributions, normalized per lepton energy
    lepton_angle_probs = np.zeros_like(weighted_xs)
    colsum = weighted_xs.sum(axis=0)
    mask = colsum > 0
    lepton_angle_probs[:, mask] = weighted_xs[:, mask] / colsum[mask]

    ########################
    ##LOAD UP NUCDEEX DATA##
    ########################
    ex_dfs = utils.loadNucDeExData(deexcitation_folder)

    ############
    #MAIN LOOP##
    ############
    #Sample neutrino eneries from folded xs times spectrum
    sampled_neutrino_energies_MeV = np.random.choice(energies_MeV_interp, size=nps, p=neutrino_energy_probs)
    items = [energies_MeV_interp,excitedLevels_MeV,excitedLevels_J,excitedLevels_parity,excitation_probs,ex_dfs,
             angles_deg_interp,lepton_angle_probs,
             neutrino_direction,utils.nuc_mass_16O_MeV,utils.nuc_mass_16F_MeV,
             "16O","16F","ve","electron"]
    args = [(Enu, *items) for Enu in sampled_neutrino_energies_MeV]

    with multiprocessing.Pool(nCPU) as pool:
      results = list(tqdm.tqdm(
        pool.imap(utils.sampleEvent,args,chunksize=200),
        total=len(args)))

    ################
    ##WRITE OUTPUT##
    ################
    outFile = open(output_name,"w")

    #Generate our events in MARLEY format
    #flux-averaged xs in natural units
    line=str(flux_averaged_xs_naturalunits)
    outFile.write(line)
    for res in results:
      header,inParticles,outParticles = res[0], res[1], res[2]
      
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