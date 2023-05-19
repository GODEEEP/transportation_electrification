# Code to Generate Medium-Duty and Heavy-Duty Electric Vehicle Load Profiles across Balancig Authorities in WECC

## Overview:  
The code generates charging profiles for medium-duty and heavy-duty vehicles across balancing authorities in WECC. To run the code, please follow the below steps:

a) Download the vehicle mobility data from NREL's Fleet DNA website (https://www.nrel.gov/transportation/fleettest-fleet-dna.html) in .csv format and save them in 'input' folder. 

b) Run 'demo_mhdv.ipynb' in 'notebooks' folder. It will call 'MHDV.py' in src folder.

c) Create 'output' folder in the same path as 'input'. The charging profiles, data (in .csv) and plots (in .png), will be avialable in 'output' folder. The profiles for Balancing Authorities are UTC time adjusted, while the profiles from FleetDNA and WECC are not. 

'depot_charging.py' in 'src' folder is modified, which is orginally created by NREL for the study, "Heavy-Duty Truck Electrification and the Impacts of Depot Charging
on Electricity Distribution Systems", by Borlaug et al., published in 2021 (website: https://www.nature.com/articles/s41560-021-00855-0). 
'Percent_mdv_wecc.py' in src folder calculates the penetration of MHDVs in some BAs from the penetration of LDVs. 
