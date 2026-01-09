# Overview
OverviewPython adaptation of [NEWTON](https://github.com/itscubist/newton), with a few changes. Still under development...

## Changes w.r.t. NEWTON:
- Ultimately will use different data-thieved cross sections/partial cross sections, supporting multiple inclusive cross sections
- Has different lepton angular distributions for supernova and pion decay-at-rest neutrinos
- Interpolates energy with more bins
- Uses a [fork](https://github.com/schedges/NucDeEx) of [NucDeEx](https://github.com/SeishoAbe/NucDeEx) for the nuclear de-excitation of the residual nucleus
- Includes nuclear recoil to conserve energy and momentum.

## To run:
* First run the python script in [this fork of NucDeEx](https://github.com/schedges/NucDeEx) to generate the de-excitation data. Then move that to the data/nucdeex folder. Then the notebook can be run to generate a MARLEY-style ascii file. 
