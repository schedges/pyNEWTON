# Functions contained in this code
# loadCrossSectionCSV(fname,threshold=15.413)
#   Description: Loads a CSV of Energy, XS, returns as numpy arrays. Removes values below threshold and puts xs=0.
#     Also adds point of threshold,0 to etsure proper interpolation
#   Input: CSV of format Energy [MeV], xs [cm2] with header of "Energy,XS"
#   Output: Numpy arrays of energy,xs
#
# loadNazakatoPartialXSData(fname) : 
#   Description: Reads the partial xs data from NEWTON (origin is Nakazato paper)
#   Input: CSV fo format Energy [MeV], J, Parity (1='-',0='+'), xs@20MeV [cm2], xs@40MeV [cm2], xs@60MeV[cm2]   
#     where Energy is excited 16F level w.r.t. to 16O ground state, i.e. offset by 14.91 MeV. Expects header 
#     of 'Energy,J,Parity,XS@20,XS@40,XS@60'. Parity required to be '1' = -, '0' = +
#   Output: Returns these np arrays in this order. Parity output as "+" or "-"
import pandas as pd
import numpy as np
import uproot as up
import os

#######################
##loadCrossSectionCSV##
#######################
def loadCrossSectionCSV(fname,threshold=15.412):
  df = pd.read_csv(fname,comment="#")

  #Check columns
  required = {"Energy", "XS"}
  if not required.issubset(df.columns):
    raise ValueError(f"File {fname} is missing required headers. Found: {list(df.columns)}")

  #Remove all data points below threshold, these are unphysical
  df = df[df["Energy"] > threshold] 

  #Insert point at threshold,xs=0.
  energies = np.array([threshold])
  xs = np.array([0])

  #Make arrays, retrun
  return np.concatenate([energies,df["Energy"].values]), np.concatenate([xs,df["XS"].values])

#############################
##loadNazakatoPartialXSData##
#############################
def loadNazakatoPartialXSData(fname,xs_threshold=1e-46):
  df = pd.read_csv(fname,comment="#")

  #Check columns
  required = {'Energy','J','Parity','XS@20','XS@40','XS@60'}
  if not required.issubset(df.columns):
    raise ValueError(f"File {fname} is missing required headers. Found: {list(df.columns)}")

  #Adjust parity: 1 = '-', 0 = '+'
  df['Parity'] = df['Parity'].map({1: '-', 0: '+'})

  #We set xs values less than threshold to 0. These are below the range shown on the plot
  df[df[["XS@20","XS@40","XS@60"]] <= xs_threshold] = 0.0

  return df['Energy'].values,df['J'].values,df['Parity'].values,df['XS@20'].values,df['XS@40'].values,df['XS@60'].values
  
#########################
##loadHaxtonMuDARAngles##
#########################
def loadHaxtonMuDARAngles(fname):
  haxton_lepton_degrees = [15.,30.,45.,60.,75.,90.,105.,120.,135.,150.,165.]
  
  cols = [[f"E@{int(i)}", str(int(i))] for i in haxton_lepton_degrees]
  cols = np.array(cols).flatten()
  df = pd.read_csv(fname, usecols=cols, comment="#")

  lepton_energies = []
  lepton_xs = []
  
  for deg in haxton_lepton_degrees:
    e_col = f"E@{int(deg)}"
    xs_col = str(int(deg))
        
    e_list = df[e_col].dropna().tolist()
    x_list = df[xs_col].dropna().tolist()
        
    lepton_energies.append(e_list)
    lepton_xs.append(x_list)

  return np.array(haxton_lepton_degrees),lepton_energies,lepton_xs

###################
##loadNucDeExData##
###################
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


############################
##loadNewtonDoubleDiffData##
############################
def loadNewtonDoubleDiffData(folderName):
  files = [folderName + i for i in os.listdir(folderName) if i.endswith(".txt")]
  files.sort()
  
  #NEWTONs data is given at cos(theta) values ranging from -0.95 to 0.95
  cos_angles_raw = np.linspace(-0.95, 0.95, 20,endpoint=True)
  angles_deg_raw = np.asarray([np.arccos(angle)*180./np.pi for angle in cos_angles_raw])
  angles_deg_raw = angles_deg_raw[::-1] #Put in increasing order
  
  Enus_raw = []
  Enus_vs_angle_raw = []

  for f in files:
    data = pd.read_csv(f, sep=r'\s+', header=None, comment="#").values

    Enus_raw.append(data[:, 0])
    xs_matrix = data[:, 1:]
    xs_matrix_flipped = np.flip(xs_matrix, axis=1)

    Enus_vs_angle_raw.append(xs_matrix_flipped.T)
  
  Enus_raw=np.asarray(Enus_raw)
  Enus_vs_angle_raw=np.asarray(Enus_vs_angle_raw)
  return angles_deg_raw, Enus_raw, Enus_vs_angle_raw