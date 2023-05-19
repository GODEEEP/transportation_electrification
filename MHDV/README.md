# Code to Generate Medium-Duty and Heavy-Duty Electric Vehicle Load Profiles across Balancig Authorities in the Western U.S. Interconnect

## Overview:  
The code generates charging profiles for medium-duty and heavy-duty vehicles across balancing authorities in the Western U.S. Interconnect. To run the code, please follow the below steps:

a) Download the vehicle mobility data from NREL's Fleet DNA website (https://www.nrel.gov/transportation/fleettest-fleet-dna.html) in .csv format and save them inside 'input' folder in the main branch. Note that you might get errors if Fleet DNA data changes. Please contact us if it happens.   

b) Run 'MHDV.ipynb' inside 'notebook' folder. It will call 'MHDV.py' in 'src' folder.

c) Create 'output' folder in the same path level as 'notebook'. The charging profiles, data (in .csv) and plots (in .png), will be avialable in the 'output' folder. The profiles for Balancing Authorities are UTC time adjusted, while the FleetDNA profiles are normalized and are local to timezones. 

'depot_charging.py' inside 'src' folder is modified, which is orginally created by NREL for the study, "Heavy-Duty Truck Electrification and the Impacts of Depot Charging
on Electricity Distribution Systems", by Borlaug et al., published in 2021 (website: https://www.nature.com/articles/s41560-021-00855-0). 

'mhdv_penetration_in_BA.ipynb' inside 'notebook' folder calculates the missing MHDV penetration in BAs using the LDV penetration.
