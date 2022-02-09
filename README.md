# Wearable sensor deployment for pandemic mitigation
Duarte N, Arora RK, Bennett G, Wang M, Snyder MP, Cooperstock JR, Wagner CE

**Description of files:**
- perform_simulation.py is used to configure parameters for a simulation, run the simulation, and plot the resulting outcome
- simulation_toolbox.py contains classes and functions used to run the simulation
- counterfactual_simulation.png is an example of a simulation's outcome

**Software and library requirements:**
- Python version 3.7.4
- pandas version 1.0.4
- matplotlib version 3.5.1
- numpy version 1.18.5
- scipy version 1.7.1

**Running simulations:**
- Install Python 3.7.4 and relevant libraries above
- Within perform_simulation.py, configure parameters:
  -  File path to the data CSV (usually the same folder as the one containing the code)
  -  Simulation start and end date (duration of N days)
  -  Number of simulations to perform of a particular scenario (recommended: 1000)
  -  Infection parameters (Supplementary Table 1; α from Supplementary Table 2)
  -  Wearable parameters (σ<sub>w</sub> and ν<sub>w</sub> from Supplementary Table 2)
  -  Policy parameters (θ, ψ, σ<sub>a</sub>, ν<sub>a</sub>, and ε from Supplementary Table 2)
  -  Economic parameters (Supplementary Table 5)

**Conducting analysis of simulation outcomes (i.e., the outcome_data variable):**
- outcome_data[0]: vector of length N with dates in simulation time frame
- outcome_data[1-4]: arrays of shape (3,N) with daily time series of outcomes and 95% confidence intervals:
  - new infections
  - number of wearable device users incorrectly quarantining
  - number of lab-based tests performed
  - number of antigen tests performed
- outcome_data[5-13]: arrays of shape (3,1) with estimate of outcomes over the simulation period and 95% confidence intervals:
  - averted infections
  - averted hospitalizations
  - number of days incorrectly spent in quarantine
  - number of days correctly spent in quarantine
  - antigen tests performed
  - lab-based tests performed
  - costs (costs of lab-based tests and antigent tests)
  - savings (savings from averted hospitalizations)
  - net savings (savings - costs)
