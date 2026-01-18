# Overview
This code follows the conceptual approach of [NEWTON](https://github.com/itscubist/newton). Where possible, digitized data files from the original NEWTON repository are reused with attribution. Credit to [Baran Bodur](https://github.com/itscubist) for coming up with this approach and digitizing data.

## Changes w.r.t. NEWTON:
- Uses a [fork](https://github.com/schedges/NucDeEx) of [NucDeEx](https://github.com/SeishoAbe/NucDeEx) for the nuclear de-excitation of the residual nucleus
- Added an option to sample directly from Haxton's plot of lepton angle vs. lepton energy. There are some small artifacts introduced using the existing angular sampling from NEWTON, but these are very minor. angular_sampling_comparison.ipynb shows this comparison.
- Apply a threshold to the partial cross sections used by NEWTON to remove points below the y-axis.
- Output format is either a ROOT TTree or [MARLEY](https://github.com/MARLEY-MC/marley)-style ascii output

## Running:
- Specify output format, number of events, and whether you want to use NEWTON's default angular distribution or one specifically taken for muDAR neutrinos. They should be nearly identical for muDAR neutrinos, but only the former will work for other spectra.
- validation.ipynb is a notebook for checking plots. pyNewton.py is faster, generates no plots, and supports multiprocessing.
- Run with the following optional command-line inputs:
```bash
python pyNewton.py [output name] [nps] [angle_sampling_type: newton | mudar]
```

## Required python packages
- numpy
- pandas
- awkward_pandas
- tqdm
- scipy
- uproot
- matplotlib (for validation plots only)
- pyROOT (for direct root output only)

## Overview of approach
1. Load data:
   - Exclusive cross section from [W. Haxton, Phys. Rev. D 36, 2283](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.36.2283). We use the data file from NEWTON, and interpolate it to the target energy grid
   - Partial cross sections for Enu=20,40,60 MeV from [K. Nakazato, et al., arxiv:1809.08398](https://arxiv.org/abs/1809.08398). Again, we use the files from NEWTON
   - Lepton opening angle vs. kinetic energy for muDAR neutrinos from [W. Haxton, Phys. Rev. C 37 2660](https://journals.aps.org/prc/abstract/10.1103/PhysRevC.37.2660). This is different than what NEWTON uses, which is based on a similar distribution for supernova neutrinos, unweighted by the spectrum shape. We interpolate this over the angle and energy grid.
   - Lepton opening angle vs. kinetic energy from [W. Haxton, Phys. Rev. D 36, 2283](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.36.2283)
   - Nuclear de-excitation data from [NucDeEx](https://github.com/SeishoAbe/NucDeEx) generated at the excited levels calculated in the Nakazato et al. paper
2. Calculate partial cross sections: Follow the approach from the Nakazato paper to fit the partial cross sections as a function of neutrino energy. We apply a threshold to the partial cross sections from NEWTON, as including some of the values it uses leads to unstable fit parameters for some exclusive cross sections. Interpolate over our energy grid.
3. Calculate the muon decay-at-rest spectrum, fold with the inclusive cross section
4. Sample events from that folded spectrum. For each event:
    - Sample the excited state using the Nakazato et al. partial cross sections.
    - If angular distribution sampling set to 'mudar':
      - Estimate the lepton energy (E_lep = E_nu - Ex - Threshold)
      - Use that energy to sample an angle for the lepton
    - Otherwise:
      - Samples from Enu vs. lepton angle plots for each Ex
    - Use kinematics to calculate an exact lepton energy. I believe this is different from standard NEWTON which ignores the nuclear recoil
    - Use that lepton energy to calculate the nuclear recoil.
    - Sample de-excitation products for this Ex using NucDeEx. Boost these by the nuclear recoil
6. Write out events

## Note:
The product of these interactions are potentially unstable nuclear isotopes. The decay of those is not handled by this generator, but can be handled with a MC simulator like Geant4
