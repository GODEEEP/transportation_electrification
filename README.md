### GODEEEP Transportation Electrification Model
Downscales GCAM transportation electrification loads from annual  state-level loads to hourly balancing authority level loads.

Different approaches are used for each class of vehicles:
* Light Duty Vehicles (LDV): utilizes NREL's [EVI-Pro-Lite API](https://developer.nrel.gov/docs/transportation/evi-pro-lite-v1/) and county meteorology to generate annual hourly load profiles aggregated to balancing authority.
* Medium and Heavy Duty Vehicles (MHDV): utilizes a modified version of NREL's [Heavy-Duty Electric Truck Depot Load](https://github.com/NREL/hdev-depot-charging-2021) model to generate daily hourly load profiles by balancing authority, extended to an annual period.
* Ship, train, and aviation vehicle loads are created by estimating county shares of each mode and population-weighting each county's contribution to balancing authorities.  

Check out the Jupyter notebooks within the sub-folders for instructions and code for running each model, and then use the notebook `combined_transportation_load.ipynb` to aggregate the results!

#### Workflow Diagram
```mermaid
flowchart LR

    linkStyle default stroke:black;
    classDef marker stroke:black,fill:black;
    classDef dataset fill:#689c73,stroke:black,stroke-width:0;
    classDef model fill:#207dcd,stroke:black,stroke-width:0;
    classDef code fill:#e88824,stroke:black,stroke-width:0;


    subgraph legend[Legend]
        dataset[(dataset)]:::dataset
        model((model)):::model
        code{{code}}:::code
    end
    
    subgraph workflow[Transportation Electrification]
        direction LR
        
        GCAM-USA((GCAM-USA)):::model
        County-Met[(County Meteorology)]:::dataset
        Combined[(Combined Transportation Loads)]:::dataset
        
        subgraph transport[ ]
            direction TB
            subgraph LDVgroup[LDV]
                direction TB
                EVI-Pro-Lite((EVI-Pro-Lite)):::model
                LDV{{LDV}}:::code
                EVI-Pro-Lite-->LDV
            end
            subgraph MHDVgroup[MHDV]
                direction TB
                MHDV{{MHDV}}:::code
            end
            subgraph OtherGroup[Ship, Train, Aviation]
                direction TB
                Other{{Other Vehicles}}:::code
            end
        end
        
        GCAM-USA-->transport
        County-Met-->transport
        transport-->Combined
    
    end
```