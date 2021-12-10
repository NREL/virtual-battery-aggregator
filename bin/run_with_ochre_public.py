import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ochre import Battery, __version__

from aggregator import BatteryAggregator

# script to run with public version of OCHRE, see https://github.com/jmaguire1/Public_OCHRE

locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator, show_offset=False)

# Parameters for simulation
time_res = dt.timedelta(minutes=5)
temperature_schedule = {}  # {'Indoor': 20}
battery_properties = {
    # Simulation parameters
    'start_time': dt.datetime(2018, 1, 1, 0, 0),  # year, month, day, hour, minute
    'time_res': time_res,
    'duration': dt.timedelta(days=5),

    # Output parameters
    # 'output_path': default_output_path,
    # 'save_results': True,
    'verbosity': 7,  # verbosity of results file (1-9)

    # Battery parameters for more, see ochre/defaults/Battery/default_parameters.csv
    'capacity': 5,  # Power capacity, kW
    'capacity_kwh': 10,  # Energy capacity, kWh
    'soc_init': 0.2,

    # Thermal parameters
    # 'zone': 'external',
    # 'initial_schedule': temperature_schedule,
}
battery_properties2 = battery_properties.copy()
battery_properties2.update({
    'soc_init': 0.5,
    'capacity': 5,  # Power capacity, kW
    'capacity_kwh': 10,  # Energy capacity, kWh
})
battery_properties3 = battery_properties.copy()
battery_properties3.update({
    'soc_init': 0.8,
    'capacity': 10,  # Power capacity, kW
    'capacity_kwh': 20,  # Energy capacity, kWh
})

# Create 3 OCHRE battery instances
print(f'Creating OCHRE (v{__version__}) Battery models')
b1 = Battery(**battery_properties)
b2 = Battery(**battery_properties2)
b3 = Battery(**battery_properties3)
battery_dict = {
    'b1': b1,
    'b2': b2,
    'b3': b3,
}


# function to collect battery properties for aggregator
def get_ochre_properties(battery: Battery, soc_min=None, soc_max=None):
    # get SOC limits
    if soc_min is None:
        soc_min = battery.soc_min
    if soc_max is None:
        soc_max = battery.soc_max

    # update SOC and battery capacity using SOC limits (converts to a SOC model with limits of [0, 1])
    capacity_kwh = battery.capacity_kwh * (soc_max - soc_min)
    soc_new = (battery.soc - soc_min) / (soc_max - soc_min)

    # get charge/discharge efficiencies at half of power capacity
    battery.electric_kw = battery.capacity / 2
    eta_charge = battery.get_efficiency()
    battery.electric_kw = -battery.capacity / 2
    eta_discharge = battery.get_efficiency()

    return {
        'State of Charge (-)': soc_new,
        'Energy Capacity (kWh)': capacity_kwh,
        'Power Capacity (kW)': battery.capacity,
        'Charge Efficiency (-)': eta_charge,
        'Discharge Efficiency (-)': eta_discharge,
    }


# Create Battery Aggregator
models = {name: get_ochre_properties(battery) for name, battery in battery_dict.items()}
agg = BatteryAggregator(time_res, models=models)

# Alternate method:
# agg = BatteryAggregator(time_res)
# for name, model in models.items():
#     agg.add_models(**{name: model})
# agg.make_opt_problem()

print('Created Aggregator with battery models:')
print(agg.models)

# ********* Start Simulation ************

powers = []
socs = []
for _ in range(36):  # 3 hours
    # Get updated parameters from OCHRE models
    models = {name: get_ochre_properties(battery) for name, battery in battery_dict.items()}

    # Send parameters to aggregator
    agg.update_models(**models)

    # Aggregate to create virtual battery model
    virtual_model = agg.aggregate()
    # print('Virtual battery model parameters:', virtual_model)

    # Send virtual model to FRS and receive virtual battery setpoint
    virtual_setpoint = 8  # in kW

    # Dispatch setpoint to individual batteries
    setpoints = agg.dispatch(p_setpoint=virtual_setpoint)
    # print(f'Setpoints for individual batteries with a virtual battery setpoint of {virtual_setpoint} kW: {setpoints}')

    # run each OCHRE battery with setpoint controls
    for name, battery in battery_dict.items():
        controls = {'P Setpoint': setpoints.get(name)}
        battery.update(1, temperature_schedule, controls)
        battery.update_model(None)
        power_to_dss = battery.electric_kw  # in kW, power output of battery to send to grid simulator

    # collect results
    powers.append({name: battery.electric_kw for name, battery in battery_dict.items()})
    socs.append({name: battery.soc for name, battery in battery_dict.items()})

# plot results
times = pd.date_range(b1.start_time, freq=b1.time_res, periods=len(powers))
powers = pd.DataFrame(powers, index=times)
socs = pd.DataFrame(socs, index=times)
# powers.plot()
# socs.plot()

fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all')
ax1.stackplot(powers.index, powers.values.T, labels=powers.columns)
ax1.axhline(8, color='k', label='Virtual Setpoint')
ax2.plot(socs.index, socs.values, label=socs.columns)
ax2.xaxis.set_major_formatter(formatter)
ax1.set_ylabel('Battery Power (kW)')
ax2.set_ylabel('Battery SOC (-)')
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles[::-1], labels[::-1], loc='lower right')
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles[::-1], labels[::-1], loc='lower right')
plt.show()
