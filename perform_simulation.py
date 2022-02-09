# Data handling libraries
import pandas as pd
import datetime

# Plotting libraries
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
from pandas.plotting import register_matplotlib_converters
pd.options.mode.chained_assignment = None  # default='warn'
register_matplotlib_converters()

# Functions for running simulations
from simulation_toolbox import Region, run_MC_simulations, extract_CI_ranges


''' =================================================================== '''
''' ========================== CONFIGURATION ========================== '''
''' =================================================================== '''

# Set file path to data
fpath = "/Users/nathanduarte/GitRepos/wearables_for_pandemic_mitigation/ihme_model_downloaded_on_20211207.csv"

# Set simulation timeframe
START_DATE = datetime.date(2020, 9, 1)
END_DATE = datetime.date(2021, 2, 20)

# Set number of simulations to run
N_SIMS = 100

# Set infection parameters
infection_params = {"LATENCY": 3.69,
                    "PRESYMP": 1.31,
                    "INFECTIOUS": 5.69,
                    "ASYMPTOMATIC": 0.40,
                    "ASYMPTOMATIC_N": 200,
                    "PRE_ASY_TRANSMISSION_RATE": 0.55,
                    "BETA_FACTOR": 1}

# Set wearable parameters
wearable_params = {"WATCH_SE": 0.80,
                   "WATCH_SE_N": 84,
                   "WATCH_SP": 0.92,
                   "WATCH_SP_N": 818}

# Set policy (i.e., behavioural, complementary intervention) parameters
policy_params = {"UPTAKE": 0.04,
                 "ADHERENCE": 0.50,
                 "ADHERENCE_N": 1723,
                 "ANTIGEN_SE": 0.911,
                 "ANTIGEN_SP": 0.997,
                 "USE_ANTIGEN": False,
                 "QUARANTINE_DAYS": 2}

# Set economic parameters
econ_params = {"HOSPITAL_COST": 23470.74,
               "ANTIGEN_COST": 5,
               "NAAT_COST": 100}


''' =================================================================== '''
''' ===================== RUN SIMULATION AND PLOT ===================== '''
''' =================================================================== '''

# Import IHME model dataset (downloaded Dec 7, 2021)
ihme_df = pd.read_csv(fpath)

# Downselect to get region-specific data within desired date range
region_df = ihme_df[ihme_df['location_name'] == "Canada"]
region_df['date'] = pd.to_datetime(region_df['date'])
region_df['date'] = region_df['date'].dt.date
region_df = region_df.set_index('date')
region_df = region_df.loc[(region_df.index > datetime.date(2020, 2, 1)) & (region_df.index < datetime.date(2021, 6, 1)), :]

# Set up colours
COLOURS_HEX = ["882255"]
COLOURS_RGB = [tuple(int(hex[i:i+2], 16)/255 for i in (0, 2, 4)) for hex in COLOURS_HEX]

# Initialize region
canada = Region("Canada", region_df)

# Perform simulations
sim_data, sim_data_d = run_MC_simulations(canada, START_DATE, END_DATE, infection_params, wearable_params, policy_params, econ_params, N_SIMS)
outcome_data = extract_CI_ranges(sim_data)

# Plot incidence of infection
fig, ax = plt.subplots(1,1, figsize=(4.5, 3), dpi=300, tight_layout=True)
plt.rcParams["font.family"] = "Roboto"
plt.rcParams["axes.titlesize"] = 14

plot_start = START_DATE - datetime.timedelta(20)
restricted_plot_dates = canada.dates[(canada.dates > plot_start) & (canada.dates <= END_DATE)]
ax.fill_between(restricted_plot_dates[restricted_plot_dates <= START_DATE], 0, 25, color='k', alpha=0.05, lw=0)
ax.axvline(datetime.date(2021, 1, 1), 0, 25, color='gray', linestyle="--", linewidth=0.75, alpha=0.75)
ax.text(datetime.date(2021, 1, 3), 1.5, '2021', size=6, color='gray', rotation=90, alpha=0.75)

restricted_nominal_curve_inf = canada.new_inf[(canada.dates > plot_start) & (canada.dates <= END_DATE)]
ax.plot(restricted_plot_dates, restricted_nominal_curve_inf/1000, color='k', linewidth=1, label="IHME Model")

ax.plot(outcome_data[0], outcome_data[1][1,:]/1000, color=COLOURS_RGB[0], linewidth=1, label="Wearable Sensor Deployment Counterfactual Scenario")
ax.fill_between(outcome_data[0], outcome_data[1][0,:]/1000, outcome_data[1][2,:]/1000, color=COLOURS_RGB[0], alpha=0.25, lw=0)

ax.set_xlim([plot_start + datetime.timedelta(1), END_DATE])
ax.set_ylim([0,25])
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.tick_params(axis = 'both', labelsize = 6)
ax.set_title('Incidence of Infection in Canada (thousands)', fontsize=8)
ax.set_xlabel('Date', fontsize=8, fontname = "Roboto")

font = font_manager.FontProperties(family="Roboto", style="normal", size=6)
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0), fancybox=False, shadow=False, ncol=2, prop=font)
figpath = "/Users/nathanduarte/GitRepos/wearables_for_pandemic_mitigation"
plt.savefig(figpath + "/counterfactual_simulation.png", bbox_inches='tight')