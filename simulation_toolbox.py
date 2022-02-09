# Data handling libraries
import numpy as np
from scipy.stats import beta
import datetime

class Region:
    
    # Initialize a region
    def __init__(self, name, dataset):
        '''
        - name is a string of the region's name
        - dataset is a time series DataFrame:
            - index is set to dates
            - incidence of infection column is called "inf_mean"
            - daily hospital admissions column is called "admis_mean"
            - daily lab-based tests performed is called "testing_mean"
        '''
        
        # Set name and population
        self.name = name
        self.N = dataset["population"].iloc[0]
        
        # Extract infections, hospitalizations, and deaths
        self.dates = dataset.index
        self.new_inf = np.asarray(dataset['inf_mean'])
        self.new_hosp = np.asarray(dataset['admis_mean'])
        self.naats_performed = np.asarray(dataset['testing_mean'])
        
        return
    

    # Estimate aggregate ratio of hospitalizations to infections
    def calculate_ihr(self, start_date, end_date, hospital_delta):
        '''
        - start_date and end_date used to denote the time period over which the aggregate ratio should be estimated
        - hospital_delta is the delay of hospitalizations relative to infections that should be used (in days)
        '''
        
        # Get baseline infections
        baseline_inf = np.sum(self.new_inf[(self.dates >= start_date) & (self.dates <= end_date)])
        
        # Get baseline hospitalizations
        ADJ_START_DATE = start_date + datetime.timedelta(hospital_delta)
        ADJ_END_DATE = end_date + datetime.timedelta(hospital_delta)
        baseline_hosp = np.sum(self.new_hosp[(self.dates >= ADJ_START_DATE) & (self.dates <= ADJ_END_DATE)])
        
        # Get IHR
        self.ihr = baseline_hosp / baseline_inf
        
        return self.ihr


    # Estimate historical transmission rate
    def extract_baseline_beta(self, infection_params):
        '''
        - format of infection_params is presented in perform_simulation.py
        '''
        
        # Unpack infection parameters
        alpha = 1 / infection_params["LATENCY"]
        tau = 1 / (infection_params["PRESYMP"])
        gamma = 1 / (infection_params["INFECTIOUS"])
        rho = infection_params["ASYMPTOMATIC"]
        lamb = infection_params["PRE_ASY_TRANSMISSION_RATE"]
        
        # Set up arrays; S, E, I, R components all represent volumes at start of day
        # Differential equations represent changes that occur during the day
        self.S_base = np.zeros(len(self.new_inf))
        self.E_base = np.zeros(len(self.new_inf))
        self.Ip_base = np.zeros(len(self.new_inf))
        self.Ia_base = np.zeros(len(self.new_inf))
        self.Is_base = np.zeros(len(self.new_inf))
        self.R_base = np.zeros(len(self.new_inf))
        self.beta = np.zeros(len(self.new_inf))
        
        # Initialize arrays; assume one person caused the first set of cases
        # The specific number of initial cases is inconsequential because we are not simulating the first wave
        self.S_base[0] = self.N - 1
        self.Ip_base[0] = 1
        
        # Move through days and calculate parameters
        for t in np.arange(0, len(self.new_inf)):
            
            # Number of new infections is known
            pi = self.new_inf[t]
            
            # Update simulation
            if (t+1) < len(self.new_inf):
                self.S_base[t+1] = self.S_base[t] - pi
                self.E_base[t+1] = self.E_base[t] + pi - (alpha*self.E_base[t])
                self.Ip_base[t+1] = self.Ip_base[t] + (alpha*self.E_base[t]) - (tau*self.Ip_base[t])
                self.Ia_base[t+1] = self.Ia_base[t] + (rho*tau*self.Ip_base[t]) - (gamma*self.Ia_base[t])
                self.Is_base[t+1] = self.Is_base[t] + ((1-rho)*tau*self.Ip_base[t]) - (gamma*self.Is_base[t])
                self.R_base[t+1] = self.R_base[t] + (gamma*self.Ia_base[t]) + (gamma*self.Is_base[t])
                
            # Calculate historical transmission rate, beta
            effective_infections = (lamb*self.Ip_base[t]) + (lamb*self.Ia_base[t]) + self.Is_base[t]
            self.beta[t] = (pi * self.N) / (self.S_base[t] * effective_infections)
            
        return
    

    # Run one counterfactual simulation
    def run_wearables_simulation(self, start_date, end_date, infection_params, wearable_params, policy_params):
        '''
        - formats of infection_params, wearable_params, and policy_params are presented in perform_simulation.py
        '''
        
        # Unpack infection parameters
        alpha = 1 / infection_params["LATENCY"]
        tau = 1 / (infection_params["PRESYMP"])
        gamma = 1 / (infection_params["INFECTIOUS"])
        kappa = 1 / (infection_params["LATENCY"] + infection_params["PRESYMP"] + infection_params["INFECTIOUS"])
        lamb = infection_params["PRE_ASY_TRANSMISSION_RATE"]
        rho = infection_params["ASYMPTOMATIC"]
        beta_factor = infection_params["BETA_FACTOR"]
        
        # Unpack wearable performance parameters
        sigma_w = wearable_params["WATCH_SE"]
        nu_w = wearable_params["WATCH_SP"]
        
        # Unpack policy parameters
        UPTAKE = policy_params["UPTAKE"]
        psi = policy_params["ADHERENCE"]
        epsilon = policy_params["QUARANTINE_DAYS"]
        if policy_params["USE_ANTIGEN"]:
            sigma_a = policy_params["ANTIGEN_SE"]
            nu_a = policy_params["ANTIGEN_SP"]
        else:
            sigma_a = 1
            nu_a = 0
        
        # Extract relevant dates and beta values
        self.dates_sim = self.dates[(self.dates >= start_date) & (self.dates <= end_date)]
        self.beta_sim = self.beta[(self.dates >= start_date) & (self.dates <= end_date)]
        
        # Set up arrays and holding variables
        dur = len(self.dates_sim)
        self.new_inf_sim = np.zeros(dur)
        self.Sw, self.Ew, self.Snw, self.Enw = np.zeros(dur), np.zeros(dur), np.zeros(dur), np.zeros(dur)
        self.Ipw, self.Iaw, self.Isw = np.zeros(dur), np.zeros(dur), np.zeros(dur)
        self.Ipnw, self.Ianw, self.Isnw = np.zeros(dur), np.zeros(dur), np.zeros(dur)
        self.Qc, self.Qi, self.R = np.zeros(dur), np.zeros(dur), np.zeros(dur)
        antigen_tests_used = 0
        self.daily_antigen_tests = np.zeros(dur)
        naats_used = 0
        self.daily_naats = np.zeros(dur)
        
        # Set up initial conditions
        self.Sw[0] = self.S_base[self.dates == start_date] * UPTAKE
        self.Snw[0] = self.S_base[self.dates == start_date] * (1-UPTAKE)
        self.Ew[0] = self.E_base[self.dates == start_date] * UPTAKE
        self.Enw[0] = self.E_base[self.dates == start_date] * (1-UPTAKE)
        self.Ipw[0] = self.Ip_base[self.dates == start_date] * UPTAKE
        self.Ipnw[0] = self.Ip_base[self.dates == start_date] * (1-UPTAKE)
        self.Iaw[0] = self.Ia_base[self.dates == start_date] * UPTAKE
        self.Ianw[0] = self.Ia_base[self.dates == start_date] * (1-UPTAKE)
        self.Isw[0] = self.Is_base[self.dates == start_date] * UPTAKE
        self.Isnw[0] = self.Is_base[self.dates == start_date] * (1-UPTAKE)
        self.R[0] = self.R_base[self.dates == start_date]
    
        # Run simulation
        for t in np.arange(0, len(self.dates_sim)):
            
            # Calculate number of new infections based on historical beta
            all_infec = np.sum([lamb * beta_factor * self.Ipw[t],
                                lamb * self.Ipnw[t],
                                lamb * beta_factor * self.Iaw[t],
                                lamb * self.Ianw[t],
                                self.Isw[t],
                                self.Isnw[t]])
            pi = self.beta_sim[t] * (self.Sw[t] + self.Snw[t]) * all_infec / self.N
            self.new_inf_sim[t] = pi
            
            # Update compartments if we are not at the end of the time frame
            if (t+1) < len(self.dates_sim):
                
                # Calculate differential equations
                self.Sw[t+1] = self.Sw[t] + np.sum([-pi * UPTAKE,
                                                    -self.Sw[t] * (1-nu_w) * (1-nu_a) * psi,
                                                    self.Qi[t] * (1/epsilon)])
                
                self.Ew[t+1] = self.Ew[t] + np.sum([pi * UPTAKE,
                                                    -self.Ew[t] * kappa * sigma_w * sigma_a * psi,
                                                    -alpha * self.Ew[t]])
                
                self.Ipw[t+1] = self.Ipw[t] + np.sum([alpha * self.Ew[t],
                                                      -self.Ipw[t] * kappa * sigma_w * sigma_a * psi,
                                                      -tau * self.Ipw[t]])
                
                self.Iaw[t+1] = self.Iaw[t] + np.sum([tau * self.Ipw[t] * rho,
                                                      -self.Iaw[t] * kappa * sigma_w * sigma_a * psi,
                                                      -gamma * self.Iaw[t]])
                
                self.Isw[t+1] = self.Isw[t] + np.sum([tau * self.Ipw[t] * (1-rho),
                                                      -gamma * self.Isw[t]])
                
                self.Snw[t+1] = self.Snw[t] + np.sum([-pi * (1-UPTAKE)])
                
                self.Enw[t+1] = self.Enw[t] + np.sum([pi * (1-UPTAKE),
                                                      -alpha * self.Enw[t]])
                
                self.Ipnw[t+1] = self.Ipnw[t] + np.sum([alpha * self.Enw[t],
                                                        -tau * self.Ipnw[t]])
                
                self.Ianw[t+1] = self.Ianw[t] + np.sum([tau * self.Ipnw[t] * rho,
                                                        -gamma * self.Ianw[t]])
                
                self.Isnw[t+1] = self.Isnw[t] + np.sum([tau * self.Ipnw[t] * (1-rho),
                                                        -gamma * self.Isnw[t]])
                
                self.Qi[t+1] = self.Qi[t] + np.sum([self.Sw[t] * (1-nu_w) * (1-nu_a) * psi,
                                                    -self.Qi[t] * (1/epsilon)])
                
                self.Qc[t+1] = self.Qc[t] + np.sum([self.Ew[t] * kappa * sigma_w * sigma_a * psi,
                                                    self.Ipw[t] * kappa * sigma_w * sigma_a * psi,
                                                    self.Iaw[t] * kappa * sigma_w * sigma_a * psi,
                                                    -self.Qc[t] * (1/epsilon)])
                
                self.R[t+1] = self.R[t] + np.sum([gamma * self.Iaw[t],
                                                  gamma * self.Isw[t],
                                                  gamma * self.Ianw[t],
                                                  gamma * self.Isnw[t],
                                                  self.Qc[t] * (1/epsilon)])
                
            # Calculate number of antigen tests used on day t
            self.daily_antigen_tests[t] = np.sum([self.Sw[t] * (1-nu_w) * psi,
                                                  self.Ew[t] * kappa * sigma_w * psi,
                                                  self.Ipw[t] * kappa * sigma_w * psi,
                                                  self.Iaw[t] * kappa * sigma_w * psi])
            antigen_tests_used = antigen_tests_used + self.daily_antigen_tests[t]
            
            # Calculate number of NAATs used on day t
            self.daily_naats[t] = np.sum([self.Sw[t] * (1-nu_w) * (1-nu_a) * psi,
                                          self.Ew[t] * kappa * sigma_w * sigma_a * psi,
                                          self.Ipw[t] * kappa * sigma_w * sigma_a * psi,
                                          self.Iaw[t] * kappa * sigma_w * sigma_a * psi])
            naats_used = naats_used + self.daily_naats[t]
                
        # Report simulation results
        self.baseline_infections = np.sum(self.new_inf[(self.dates >= start_date) & (self.dates <= end_date)])
        self.averted_infections = self.baseline_infections - np.sum(self.new_inf_sim)
        self.averted_hospitalizations = self.averted_infections * self.calculate_ihr(start_date, end_date, 7)
        self.incorrect_quarantines = np.sum(self.Qi)
        self.correct_quarantines = np.sum(self.Qc)
        if policy_params["USE_ANTIGEN"]:
            self.antigen_tests = antigen_tests_used
        else:
            self.antigen_tests = 0
            self.daily_antigen_tests = np.zeros(dur)
        self.naats = naats_used
        
        return

    
# Perform Monte Carlo simulations
def run_MC_simulations(region, START_DATE, END_DATE, infection_params, wearable_params, policy_params, econ_params, n_sims, uptake_spend=0):
    '''
    - formats of infection_params, wearable_params, and policy_params are presented in perform_simulation.py
    - n_sims is the number of simulations to run
    '''
        
    # Unpack random variable parameters to keep a record of baseline settings
    WATCH_SE = wearable_params["WATCH_SE"]
    WATCH_SE_N = wearable_params["WATCH_SE_N"]
    WATCH_SP = wearable_params["WATCH_SP"]
    WATCH_SP_N = wearable_params["WATCH_SP_N"]
    ADHERENCE = policy_params["ADHERENCE"]
    ADHERENCE_N = policy_params["ADHERENCE_N"]
    ASYMPTOMATIC = infection_params["ASYMPTOMATIC"]
    ASYMPTOMATIC_N = infection_params["ASYMPTOMATIC_N"]
    
    # Identify length of simulation
    dur = (END_DATE - START_DATE).days + 1
    
    # Create holding arrays
    new_inf_holding = np.zeros((n_sims, dur))
    Qi_holding = np.zeros((n_sims, dur))
    averted_inf_holding = np.zeros(n_sims)
    averted_hosp_holding = np.zeros(n_sims)
    incorrect_quarantines_holding = np.zeros(n_sims)
    correct_quarantines_holding = np.zeros(n_sims)
    antigen_tests_holding = np.zeros(n_sims)
    daily_antigen_tests_holding = np.zeros((n_sims, dur))
    naats_holding = np.zeros(n_sims)
    daily_naats_holding = np.zeros((n_sims, dur))
    spending_holding = np.zeros(n_sims)
    savings_holding = np.zeros(n_sims)
    net_savings_holding = np.zeros(n_sims)
    
    # Run desired number of simulations
    for i in np.arange(0, n_sims):
        
        # Draw random variables from distributions; drawn values will remain consistent through each simulation
        wearable_params["WATCH_SE"] = beta.rvs(WATCH_SE*WATCH_SE_N, (1-WATCH_SE)*WATCH_SE_N)
        wearable_params["WATCH_SP"] = beta.rvs(WATCH_SP*WATCH_SP_N, (1-WATCH_SP)*WATCH_SP_N)
        policy_params["ADHERENCE"] = beta.rvs(ADHERENCE*ADHERENCE_N, (1-ADHERENCE)*ADHERENCE_N)
        infection_params["ASYMPTOMATIC"] = beta.rvs(ASYMPTOMATIC*ASYMPTOMATIC_N, (1-ASYMPTOMATIC)*ASYMPTOMATIC_N)
        
        # Run simulation
        region.extract_baseline_beta(infection_params)
        region.run_wearables_simulation(START_DATE, END_DATE, infection_params, wearable_params, policy_params)
        
        # Store simulation-based results
        new_inf_holding[i,:] = region.new_inf_sim
        Qi_holding[i,:] = region.Qi
        averted_inf_holding[i] = region.averted_infections
        averted_hosp_holding[i] = region.averted_hospitalizations
        incorrect_quarantines_holding[i] = region.incorrect_quarantines
        correct_quarantines_holding[i] = region.correct_quarantines
        antigen_tests_holding[i] = region.antigen_tests
        daily_antigen_tests_holding[i,:] = region.daily_antigen_tests
        naats_holding[i] = region.naats
        daily_naats_holding[i,:] = region.daily_naats
        
        # Calculate economic outcomes
        spending_holding[i] = np.sum([naats_holding[i] * econ_params["NAAT_COST"],
                                      antigen_tests_holding[i] * econ_params["ANTIGEN_COST"]])
        savings_holding[i] = averted_hosp_holding[i] * econ_params["HOSPITAL_COST"]
        net_savings_holding[i] = savings_holding[i] - spending_holding[i]

    # Package results from simulations
    sim_data = [region.dates_sim, new_inf_holding, Qi_holding, daily_naats_holding, daily_antigen_tests_holding,
                averted_inf_holding, averted_hosp_holding, incorrect_quarantines_holding,
                correct_quarantines_holding, antigen_tests_holding, naats_holding, spending_holding, savings_holding,
                net_savings_holding]
    
    sim_data_downsampled = [averted_inf_holding[np.arange(0,n_sims,10)],
                            averted_hosp_holding[np.arange(0,n_sims,10)],
                            incorrect_quarantines_holding[np.arange(0,n_sims,10)],
                            correct_quarantines_holding[np.arange(0,n_sims,10)],
                            antigen_tests_holding[np.arange(0,n_sims,10)],
                            naats_holding[np.arange(0,n_sims,10)],
                            spending_holding[np.arange(0,n_sims,10)],
                            savings_holding[np.arange(0,n_sims,10)],
                            net_savings_holding[np.arange(0,n_sims,10)]]
    
    # Reset random variables to baseline
    wearable_params["WATCH_SE"] = WATCH_SE
    wearable_params["WATCH_SE_N"] = WATCH_SE_N
    wearable_params["WATCH_SP"] = WATCH_SP
    wearable_params["WATCH_SP_N"] = WATCH_SP_N
    policy_params["ADHERENCE"] = ADHERENCE
    policy_params["ADHERENCE_N"] = ADHERENCE_N
    infection_params["ASYMPTOMATIC"] = ASYMPTOMATIC
    infection_params["ASYMPTOMATIC_N"] = ASYMPTOMATIC_N
    
    return sim_data, sim_data_downsampled


# Extract confidence intervals from Monte Carlo simulation data
def extract_CI_ranges(sim_data, lower=2.5, upper=97.5):
    
    # Create holding array for outcome summary
    outcomes = []
    
    # Add the dates row
    outcomes.append(sim_data[0])
    
    # Add the time series outcomes
    for i in np.arange(1,5):
        outcome_var_lower = np.percentile(sim_data[i], lower, axis=0)
        outcome_var_mean = np.mean(sim_data[i], axis=0)
        outcome_var_upper = np.percentile(sim_data[i], upper, axis=0)
        
        outcomes.append(np.concatenate([outcome_var_lower.reshape((1,-1)),
                                        outcome_var_mean.reshape((1,-1)),
                                        outcome_var_upper.reshape((1,-1))], axis=0))
        
    # Add the other outcomes
    for i in np.arange(5, len(sim_data)):
        outcomes.append([np.percentile(sim_data[i], lower),
                         np.mean(sim_data[i]),
                         np.percentile(sim_data[i], upper)])
        
    return outcomes