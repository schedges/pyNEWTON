#Constants and functions used by pyNEWTON
#
# Functions:
#   getNameFromPDG(pdgCode) : using the pdgDict defined here, returns the name of the particle from the PDG code
#   loadCrossSectionCSV(fname) : Loads a CSV of format Energy [MeV], xs [cm2] into np arrays and returns
#   loadNazakatoPartialXSData(fname) : Reads the partial xs data from NEWTON. Format is 
#       Energy [MeV], J, Parity (1='-',0='+'), xs@20MeV [cm2], xs@40MeV [cm2], xs@60MeV[cm2]
#       where Energy is excited 16F level w.r.t. to 16O ground state, i.e. offset by 14.91 MeV. 
#       Returns these np arrays in this order.
#   solveNakazatoCoeffs(levels,xs1,xs2,xs3,E1=20,E2=40,E3=60,threshold=1e-46): Solves the equations from the nakazato paper
#       to get a functional form of the cross section. We set xs less than 1e-46 to 0 as some of the values
#       in the input data are speculative, leading to odd fit parameters. Returns coefficients.   
#   calcNakazatoPartialXS(levels,energies,c1s,c2s,c3s): Calculates the partial cross sections of each level on our energies
#       grid, where c1s, c2s, and c3 are coefficients solved above for each level.
#   loadHaxtonAngles(fname): Loads the curves from Haxton plot of energy vs. cross section for different neutrino-lepton
#       opening angles.
#   interpolateHaxtonAngles(angles_raw,angles_interp,lepton_energies_raw,energies_interp,lepton_xs_raw):
#       Takes arrays of raw angles, raw energies from the haxton data, along with our grids to interpolate over, and 
#       does the interpolation. Assumes the xs distributions at 0-deg and 180-deg match up with the 15-deg and 165-deg
#       shapes respectively.
#   loadNucDeExData(foldername): Loads up NucDeEx root trees as pd data frames, assuming a specific filename and root format
#   calcMomentum(E, m, direction): Given an energy, mass, and direction, calculate px,py,pz
#   calcLeptonEnergy(E_nu,theta_rad,Ex,M_tar=nuc_mass_16O_MeV,M_res=nuc_mass_16F_MeV,M_lep=mass_e_MeV) :
#       Given a neutrino energy, opening angle between lepton and neutrino, an excitation level, and appropriate masses,
#       calculate the energy of the lepton
#   rotateLeptonToLabFrame
#   sampleEvent
import os
import sys
import numpy as np
import uproot as up
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

#############
##Constants##
#############
hbar_c_squared = 3.89379290*np.power(10.,-22)
amu_to_MeV = 931.49432
#AME 20 mass evaluations: https://www.anl.gov/sites/www/files/2021-05/mass_1.mas20.txt
atomic_mass_16O_amu = 15994914.61926e-6 #includes 8 electrons
atomic_mass_16F_amu = 16011460.278e-6 #includes 9 electrons
mass_16O_MeV = atomic_mass_16O_amu * amu_to_MeV
mass_16F_MeV = atomic_mass_16F_amu * amu_to_MeV
mass_e_MeV = 0.51099895069
mass_mu_amu = 113428.9267e-6
mass_mu_MeV = mass_mu_amu * amu_to_MeV
nuc_mass_16O_MeV = mass_16O_MeV - 8*mass_e_MeV
nuc_mass_16F_MeV = mass_16F_MeV - 9*mass_e_MeV
gndstate_mass_difference_MeV = 14.91
threshold = (atomic_mass_16F_amu-atomic_mass_16O_amu)*amu_to_MeV

pdg_dict = {
    "ve": {"pdg":12,"charge":0},
    "electron": {"pdg":11,"charge":-1},
    "gamma": {"pdg":22,"charge":0},
    "neutron": {"pdg":2112,"charge":0},
    "proton": {"pdg":2212,"charge":1},
    "deuteron": {"pdg":1000010020,"charge":1},
    "triton": {"pdg":1000010030,"charge":1},
    "3He": {"pdg":1000020030,"charge":2},
    "4He": {"pdg":1000020040,"charge":2},
    "11C": {"pdg":1000060110,"charge":0},
    "12C": {"pdg":1000060120,"charge":0},
    "13C": {"pdg":1000060130,"charge":0},
    "12N": {"pdg":1000070120,"charge":0},
    "13N": {"pdg":1000070130,"charge":0},
    "14N": {"pdg":1000070140,"charge":0},
    "14O": {"pdg":1000080140,"charge":0},
    "15O": {"pdg":1000080150,"charge":0},
    "16O": {"pdg":1000080160,"charge":0},
    "15F": {"pdg":1000090150,"charge":0},
    "16F": {"pdg":1000090160,"charge":0}
}

#############
##Functions##
#############
#Using the pdgDict defined above, returns the name of the particle from the PDG code
def getNameFromPDG(pdgCode):
    for item in pdg_dict.keys():
        if pdg_dict[item]["pdg"]==pdgCode:
            return item
    print(f"{pdgCode} not yet defined in dict!")
    sys.exit()

#Loads a CSV of format Energy [MeV], xs [cm2] into np arrays and returns
#'#' used to define comments
def loadCrossSectionCSV(fname):
  energies_MeV_raw = []
  xs_cm2_raw = []
  for line in open(fname,"r"):
    if not line.startswith("#"):
      line=line.strip("\n")
      if not line=="":
        lineParts = line.split(",")
        if len(lineParts)==2:
          energies_MeV_raw.append(float(lineParts[0]))
          xs_cm2_raw.append(float(lineParts[1]))
  return np.array(energies_MeV_raw),np.array(xs_cm2_raw)

# Reads the partial xs data from NEWTON. Format is 
#   Energy [MeV], J, Parity (1='-',0='+'), xs@20MeV [cm2], xs@40MeV [cm2], xs@60MeV[cm2]
# where Energy is excited 16F level w.r.t. to 16O ground state, i.e. offset by 14.91 MeV. 
# Returns these np arrays in this order.
def loadNazakatoPartialXSData(fname):
  excitedLevels_MeV = []
  excitedLevels_J = []
  excitedLevels_parity = []
  excitedXS_20MeV = []
  excitedXS_40MeV = []
  excitedXS_60MeV = []
  for line in open(fname,"r"):
    if not line.startswith("#"):
      line=line.strip("\n")
      if not line=="":
        lineParts = line.split(",")
        if len(lineParts)==6:
          excitedLevels_MeV.append(float(lineParts[0]))
          excitedLevels_J.append(int(lineParts[1]))
          if lineParts[2]=="1":
            parity = "-"
          else:
            parity = "+"
          excitedLevels_parity.append(parity)
          excitedXS_20MeV.append(float(lineParts[3]))
          excitedXS_40MeV.append(float(lineParts[4]))
          excitedXS_60MeV.append(float(lineParts[5]))
  
  #FIT COEFFICIENTS
  excitedLevels_MeV = np.asarray(excitedLevels_MeV)
  excitedXS_20MeV = np.asarray(excitedXS_20MeV)
  excitedXS_40MeV = np.asarray(excitedXS_40MeV)
  excitedXS_60MeV = np.asarray(excitedXS_60MeV)

  return np.array(excitedLevels_MeV),np.array(excitedLevels_J),np.array(excitedLevels_parity),np.array(excitedXS_20MeV),np.array(excitedXS_40MeV),np.array(excitedXS_60MeV)

# Solves the equations from the nakazato paper to get a functional form of the cross section. 
# We set xs less than 1e-46 to 0 as some of the values in the input data are speculative, 
# leading to odd fit parameters. Returns coefficients. Again, levels here defined w.r.t. 16O ground state. 
def solveNakazatoCoeffs(levels,xs1,xs2,xs3,E1=20.,E2=40.,E3=60.,threshold=1e-46):
  c1s = []
  c2s = []
  c3s = []

  #Calculate the lambda parameters for each level, provided the level is less than the neutrino energy i.e. it can be populated
  #Lambda = log10(E_nu^0.25 - Ex^0.25)
  with np.errstate(divide='ignore', invalid='ignore'):
    def calcLambda(E, Ex):
      diff = np.power(E, 0.25) - np.power(Ex, 0.25)
      return np.where(diff > 0, np.log10(diff), 0)

    lambda1 = calcLambda(E1, levels)
    lambda2 = calcLambda(E2, levels)
    lambda3 = calcLambda(E3, levels)

    #Calculate log of the xs values, replacing 0 with -np.ing
    log_xs1 = np.where(xs1 > threshold, np.log10(xs1), -np.inf)
    log_xs2 = np.where(xs2 > 0, np.log10(xs2), -np.inf)
    log_xs3 = np.where(xs3 > 0, np.log10(xs3), -np.inf)

  #Step through levels, solving system of equations for coefficients
  for i,_ in enumerate(levels):
    #If xs is 0 at 20 MeV, do a 2-parameter fit
    if log_xs1[i]==-np.inf:
      A = np.array([[1.0,lambda2[i]],
                    [1.0,lambda3[i]]])
      y = np.array([log_xs2[i],log_xs3[i]])
      a,b = np.linalg.solve(A,y)
      c = 0
    #Otherwise do a 3-parameter fit
    else:
      A = np.array([[1.0,lambda1[i],np.power(lambda1[i],2)],
                  [1.0,lambda2[i],np.power(lambda2[i],2)],
                  [1.0,lambda3[i],np.power(lambda3[i],2)]])
      y = np.array([log_xs1[i],log_xs2[i],log_xs3[i]])
      a,b,c = np.linalg.solve(A,y)

    c1s.append(a)
    c2s.append(b)
    c3s.append(c)

  return c1s,c2s,c3s

#Using the solved coefficients, determine the partial cross sections of each excited level
def calcNakazatoPartialXS(levels,energies,c1s,c2s,c3s):
  partial_xs = []

  for ilev,Ex_MeV in enumerate(levels):
    xs = np.zeros_like(energies)
    if c3s[ilev]==0:
      for inrg,E in enumerate(energies):
        if E<=Ex_MeV + mass_e_MeV: #Kinematic requirement
          continue
        Lam = np.log10(np.power(E,0.25)-np.power(Ex_MeV,0.25))
        log10XS = c1s[ilev] + c2s[ilev]*Lam
        xs[inrg] = np.power(10,log10XS)
    else:
      for inrg,E in enumerate(energies):
        if E<=Ex_MeV + mass_e_MeV: #Kinematic requirements
          continue
        Lam = np.log10(np.power(E,0.25)-np.power(Ex_MeV,0.25))
        log10XS = c1s[ilev] + c2s[ilev]*Lam + c3s[ilev]*Lam*Lam
        xs[inrg] = np.power(10,log10XS)
    
    partial_xs.append(xs)
  return partial_xs

def loadHaxtonAngles(fname):
  haxton_lepton_degrees = [15.,30.,45.,60.,75.,90.,105.,120.,135.,150.,165.]
  
  lepton_energies_raw = [[] for _ in haxton_lepton_degrees]
  lepton_xs_raw       = [[] for _ in haxton_lepton_degrees]
  
  for line in open(fname,"r"):
    if line.startswith("#"):
      continue
    line=line.strip("\n")
    if not line=="":
      lineParts = line.split(",")

      for i in range(len(haxton_lepton_degrees)):
        lepton_energy_str  = lineParts[2*i]
        lepton_xs_str =  lineParts[2*i + 1]

        if lepton_energy_str != "":
          lepton_energies_raw[i].append(float(lepton_energy_str))
          lepton_xs_raw[i].append(float(lepton_xs_str))
  return haxton_lepton_degrees,lepton_energies_raw,lepton_xs_raw

def interpolateHaxtonAngles(angles_raw,angles_interp,lepton_energies_raw,energies_interp,lepton_xs_raw):
  #Use 15 deg dist for 0, 165 deg dist for 180
  angles_tmp = angles_raw.copy()
  lepton_energies_tmp = lepton_energies_raw.copy()
  lepton_xs_tmp = lepton_xs_raw.copy()
  angles_tmp.insert(0,0.)
  lepton_energies_tmp.insert(0,list(lepton_energies_raw[0]))
  lepton_xs_tmp.insert(0,list(lepton_xs_raw[0]))
  angles_tmp.append(180.)
  lepton_energies_tmp.append(list(lepton_energies_raw[-1]))
  lepton_xs_tmp.append(list(lepton_xs_raw[-1]))

  #Interpolate lepton energies over each angle
  lepton_xs_interp = []
  for ideg,deg in enumerate(angles_tmp):
    lepton_xs_interp.append(np.interp(energies_interp,lepton_energies_tmp[ideg],lepton_xs_tmp[ideg],left=0,right=0))

  #For interpolator for interpolating angles, with raw angles and interpolated energy as the axis.
  lepton_xs_interp = np.array(lepton_xs_interp)
  grid_interp = RegularGridInterpolator(
      (np.array(angles_tmp), energies_interp),
      lepton_xs_interp,
      bounds_error=False,
      fill_value=0.0   # for energy outside range
  )

  #Use that to interpolate over our angles grid.
  A, E = np.meshgrid(angles_interp, energies_interp, indexing="ij")
  pts = np.stack([A.ravel(), E.ravel()], axis=-1)
  angle_energy_xs_cm2 = grid_interp(pts, method="linear").reshape(len(angles_interp), len(energies_interp))
  return angle_energy_xs_cm2

#Loads up NucDeEx root trees as pd data frames, assuming a specific filename and root format
def loadNucDeExData(folderName):
    ex_filenames = [i for i in os.listdir(folderName) if i.endswith(".root")]
    ex_filenames.sort()
    ex_levels = [round(float(i.split("_")[1]),2) for i in ex_filenames]
    ex_levels.insert(0,0)
    ex_dfs = [None]

    for fname in ex_filenames:
        fpath = folderName+fname
        with up.open(fpath) as f:
            # If there's only one TTree in the file
            tree = f["tree"]

            # Load all branches into a DataFrame
            df = tree.arrays(library="pd")

        ex_dfs.append(df)
        
    return ex_dfs

#Given an energy, mass, and direction, calculate px,py,pz
def calcMomentum(E, m, direction):
    direction = np.asarray(direction, dtype=float)
    direction_hat = direction / np.linalg.norm(direction)

    p = np.sqrt(E**2 - m**2)

    px, py, pz = p * direction_hat
    return px, py, pz

#Given a neutrino energy, opening angle between lepton and neutrino, an excitation level, and appropriate masses,
#calculate the energy of the lepton
def calcLeptonEnergy(E_nu,theta_rad,Ex,M_tar=nuc_mass_16O_MeV,M_res=nuc_mass_16F_MeV,M_lep=mass_e_MeV):
    #Energy conservation:
    #E_nu + M_tar = E_lep + E_res
    #with E_res = sqrt(k_res^2 + (M_res + Ex)^2)
    #Rearrange
    #E_nu + M_tar - E_lep = sqrt(k_res^2 + (M_res + Ex)^2)
    #Square both sides
    #(E_nu + M_tar - E_lep)^2 = k_res^2 + (M_res + Ex)^2
    #Solve for k_res^2
    #k_res^2 = (E_nu + M_tar - E_lep)^2 - (M_res + Ex)^2
    #To simplify, define
    var1 = E_nu + M_tar
    var2 = np.power(M_res + Ex,2)
    #k_res^2 = (var1 - E_lep)^2 - var2

    #Momentum conservation
    #k_nu = k_lep + k_res
    #rearranging
    #k_res = k_nu - k_lep
    #Calculate the magnitude
    #k_res^2 = k_nu^2 + k_lep^2 - 2*k_nu*k_lep*cos(theta)
    #Plug in k_res^2
    #(var1 - E_lep)^2 - var2 = k_nu^2 + k_lep^2 - 2*k_nu*k_lep*cos(theta)
    #But k_lep is defined as
    #k_lep = sqrt(E_lep^2 - m_lep^2)
    #and
    #|k_nu| = |E_nu|
    #So
    #(var_1 - E_lep)^2 - var2 = E_nu^2 + E_lep^2 - m_lep^2 - 2 * E_nu * sqrt(E_lep^2 - m_lep^2)*cos(theta)
    #simplify
    var3 = np.power(E_nu,2) - np.power(M_lep,2)
    var4 = 2 * E_nu * np.cos(theta_rad)
    #Equation simplifies if cosTheta is ~ 0
    if abs(np.cos(theta_rad))<1e-12:
       #(var1 - E_lep)^2 - var2 = E_nu^2 + E_lep^2 - m_lep^2
       #var1^2 + E_lep^2 - 2*var1*E_lep - var2 = E_nu^2 + E_lep^2 - m_lep^2
       #var1^2 - var2 - E_nu^2 + m_lep^2 = 2*var1*E_lep
       #E_lep = (var1^2 - var2 - E_nu^2 + m_lep^2)/(2*var1)
       E_lep = (np.power(var1,2) - var2 - np.power(E_nu,2) + np.power(M_lep,2))/(2*var1)
       return E_lep
    #to get
    #(var1 - E_lep)^2 - var2 = var3 + E_lep^2 - var4*sqrt(E_lep^2 - m_lep^2)
    #Isolate sqrt
    #var4 * sqrt(E_lep^2 - m_lep^2) = var3 + E_lep^2 + var2 - (var1 - E_lep)^2 
    #                               = var3 + var2 - var1^2 + E_lep^2 - E_lep^2 + 2*var1*E_lep
    #                               = var3 + var2 - var1^2 + 2*var1*E_lep
    #sqrt(E_lep^2 - m_lep^2) = (var3 + var2 - var1^2 + 2*var1*E_lep)/var4
    #Simplifying
    #sqrt(E_lep^2 - m_lep^2) = (var3 + var2 - var1^2)/var4 + 2*var1*E_lep/var4
    #and then defining
    var5 = (var3 + var2 - np.power(var1,2))/var4
    var6 = 2*var1/var4
    #We get
    #sqrt(E_lep^2 - m_lep^2) = var5 + var6*E_lep #CONDITION TO CHECK AT THE END
    #Squaring
    #E_lep^2 - m_lep^2 = var5^2 + var6^2*E_lep^2 + 2*var5*var6*E_lep 
    #Group
    #E_lep^2(1-var6^2) - 2*var5*var6*E_lep - var5^2 - m_lep^2 = 0
    A = 1-np.power(var6,2)
    B = -2*var5*var6
    C = - np.power(var5,2) - np.power(M_lep,2)
    #Quadratic formulate
    if B*B < 4*A*C:
        print("ERROR!")
        print(B*B,4*A*C)
        print(E_nu,theta_rad*180./np.pi,Ex)
        sys.exit()
    
    #TOTAL ENERGY
    E_lep_1 = (-B + np.sqrt(B*B - 4*A*C)) / (2*A)
    E_lep_2 = (-B - np.sqrt(B*B - 4*A*C)) / (2*A)

    #Check for validity
    sol1_is_valid = False
    sol2_is_valid = False
    if E_lep_1 > M_lep and var5 + var6*E_lep_1 > 0:
        sol1_is_valid=True
    if E_lep_2 > M_lep and var5 + var6*E_lep_2 > 0:
        sol2_is_valid=True

    if sol1_is_valid and sol2_is_valid:
        print(E_lep_1,E_lep_2)
        sys.exit()
    elif sol1_is_valid:
        E_lep = E_lep_1
    elif sol2_is_valid:
        E_lep = E_lep_2
    else:
        print(E_nu,theta_rad,Ex,E_lep_1,E_lep_2)
        print("No sol is valid")
        sys.exit()

    return E_lep

def rotateLeptonToLabFrame(nu_dir, theta, phi):
    nu_dir = np.asarray(nu_dir) / np.linalg.norm(nu_dir)

    # pick vector not parallel to nu to define phi=0 reference
    a = np.array([0.0, 0.0, 1.0])
    if abs(nu_dir[2]) > 0.999:
        a = np.array([0.0, 1.0, 0.0])

    # transverse basis t1,t2 around nu
    t1 = np.cross(a, nu_dir)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(nu_dir, t1)

    # rotation axis (unit) perpendicular to nu, at azimuth phi
    k = np.cos(phi)*t1 + np.sin(phi)*t2

    # Rodrigues: rotate nu by theta about k
    l = nu_dir*np.cos(theta) + np.cross(k, nu_dir)*np.sin(theta) + k*np.dot(k, nu_dir)*(1.0 - np.cos(theta))

    return l / np.linalg.norm(l)

#################################################
#MAIN LOOP - pass in a smapled neutrino energy##
#################################################
def sampleEvent(args):
  Enu=args[0]
  energies_MeV_interp=args[1]
  excitation_levels_MeV=args[2]
  excitation_levels_J=args[3]
  excited_levels_parity=args[4]
  excitation_probs=args[5]
  ex_dfs=args[6]
  angles_deg_interp=args[7]
  lepton_angle_probs=args[8]
  neutrino_direction=args[9]
  target_nuc_mass=args[10]
  res_nuc_mass=args[11]
  targetName=args[12]
  resName=args[13]
  neutrinoName=args[14]
  leptonName=args[15]
  
  outParticles = []
  inParticles = []

  #Get the indices on our sampled energy grid
  energy_idx = np.searchsorted(energies_MeV_interp,Enu)

  #Sample excitation state index
  ex_idx = np.random.choice([i for i in range(0,len(excitation_levels_MeV))],size=1,p=excitation_probs[:,energy_idx])[0]

  # get the excitation energy, twoJ, parity
  event_excitation_energy_MeV = round(excitation_levels_MeV[ex_idx] - gndstate_mass_difference_MeV,2) #Relative to 16F gnd!!!
  event_excitation_twoJ = int(2*excitation_levels_J[ex_idx])
  event_excitation_parity = excited_levels_parity[ex_idx]

  header={"Ex":event_excitation_energy_MeV,
          "twoJ":event_excitation_twoJ,
          "parity":event_excitation_parity}

  #Sample NucDeEx deexcitation
  #Ground state should have no nuclear de-excitation, ignore
  if not ex_idx==0:
    sampled_deex_idx = np.random.randint(0, len(ex_dfs[ex_idx]))
    event_deex = ex_dfs[ex_idx].iloc[sampled_deex_idx]
    #Check for a mismatch between Ex levels from Nakazato and from NucDeEx run
    if not event_excitation_energy_MeV==event_deex.Ex_MeV:
      print("Error sampling deex files!")
      print(ex_idx,event_deex.Ex_MeV,event_excitation_energy_MeV)
      sys.exit()

  #########
  ##INPUT##
  #########
  #Generate input particles line
  #Incident neutrino
  Enu = energies_MeV_interp[energy_idx]
  PXnu,PYnu,PZnu = calcMomentum(Enu,0,neutrino_direction)
  p_nu_vec = np.array([PXnu, PYnu, PZnu])
  inputParticles = []
  inParticles.append({"pdg":pdg_dict[neutrinoName]["pdg"],
                      "totalE":Enu,
                      "PX":PXnu,
                      "PY":PYnu,
                      "PZ":PZnu,
                      "mass":0,
                      "charge":0})
  inParticles.append({"pdg":pdg_dict[targetName]["pdg"],
                      "totalE":target_nuc_mass,
                      "PX":0,
                      "PY":0,
                      "PZ":0,
                      "mass":target_nuc_mass,
                      "charge":0})
  
  ##########
  ##OUTPUT##
  ##########
  #Calculate lepton kinematics
  #Estimate lepton energy (get approximate value)
  Ee_sampled = energies_MeV_interp[energy_idx] - event_excitation_energy_MeV - threshold
  Ee_idx = np.searchsorted(energies_MeV_interp,Ee_sampled)
  header["Ee_sampled"] = Ee_sampled

  #Use approximate lepton energy to sample lepton angle
  event_lepton_theta_deg = np.random.choice(angles_deg_interp,p=lepton_angle_probs[:,Ee_idx])
  header["theta_deg"] = event_lepton_theta_deg
  event_lepton_theta_rad = event_lepton_theta_deg*np.pi/180.
  #Calculate lepton energy
  Ee_MeV = calcLeptonEnergy(Enu, event_lepton_theta_rad, event_excitation_energy_MeV,M_tar=target_nuc_mass,M_res=res_nuc_mass)
  #Pick phi angle, convert to lab frame
  event_lepton_phi_rad = 2*np.pi*np.random.random() #Sample phi from uniform distribution
  event_lepton_direction = rotateLeptonToLabFrame(neutrino_direction,event_lepton_theta_rad,event_lepton_phi_rad)
  #Calculate lepton energy, residual total energy, residual momentum, ensuring energy and momentum are conserved
  #Calculate lepton momentum
  PXe,PYe,PZe = calcMomentum(Ee_MeV,mass_e_MeV,event_lepton_direction)
  p_lep_vec = np.array([PXe,PYe,PZe])
  outParticles.append({"pdg":pdg_dict[leptonName]["pdg"],
                       "totalE":Ee_MeV,
                       "PX":PXe,
                       "PY":PYe,
                       "PZ":PZe,
                       "mass":mass_e_MeV,
                       "charge":pdg_dict[leptonName]["charge"]
  })

  #Determine Nuclear Recoil Vector (Conservation: p_nu = p_lep + p_res)
  Eres_MeV = (Enu + target_nuc_mass) - Ee_MeV
  p_res_vec = p_nu_vec - p_lep_vec
  PXres, PYres, PZres = p_res_vec
  #Define boost vector of recoiling nucleus (beta = p/E)
  beta_vec = np.array([PXres, PYres, PZres]) / Eres_MeV
  beta_mag2 = np.dot(beta_vec, beta_vec)
  #gamma = 1.0 / np.sqrt(max(1e-12, 1.0 - beta_mag2)) #TODO: Not sure this catch is actually needed
  gamma = 1./np.sqrt(1.-beta_mag2)

  #Process De-excitation Products
  if ex_idx == 0:
      # Ground state: Output the 16F nucleus recoil
     outParticles.append({"pdg":pdg_dict[resName]["pdg"],
                         "totalE":Eres_MeV,
                         "PX":PXres,
                         "PY":PYres,
                         "PZ":PZres,
                         "mass":res_nuc_mass,
                         "charge":pdg_dict[resName]["charge"]})
  else:
      # Boost each precomputed particle from the rest frame to the lab frame
      for i in range(int(event_deex['size'])):
          pdg_i  = int(event_deex['PDG'][i])
          m_i    = event_deex['mass_MeV'][i]
          E_rest = event_deex['totalE_MeV'][i]
          p_rest = np.array([
              event_deex['PX_MeV'][i],
              event_deex['PY_MeV'][i],
              event_deex['PZ_MeV'][i]
          ])
          particleName = getNameFromPDG(pdg_i)
          charge = pdg_dict[particleName]["charge"]

          # Lorentz Boost: E_lab = gamma * (E_rest + beta . p_rest)
          E_lab = gamma * (E_rest + np.dot(beta_vec, p_rest))
          
          #Calculate momentum in lab frame
          p_lab = p_rest + ((gamma - 1) * np.dot(p_rest, beta_vec) / beta_mag2 + gamma * E_rest) * beta_vec

          outParticles.append({"pdg":pdg_dict[particleName]["pdg"],
                              "totalE":E_lab,
                              "PX":p_lab[0],
                              "PY":p_lab[1],
                              "PZ":p_lab[2],
                              "mass":m_i,
                              "charge":charge})
          
  return header,inParticles,outParticles
