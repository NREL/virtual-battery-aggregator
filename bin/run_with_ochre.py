import datetime as dt
from ochre import Battery, __version__

from aggregator import BatteryAggregator

# Parameters for simulation
time_res = dt.timedelta(minutes=5)
temperature_schedule = {'ambient_dry_bulb': 20}
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
    'soc_init': 0.5,

    # Thermal parameters
    'zone': 'external',
    'initial_schedule': temperature_schedule,
}
battery_properties2 = battery_properties.copy()
battery_properties2.update({
    'capacity': 10,  # Power capacity, kW
    'capacity_kwh': 20,  # Energy capacity, kWh
})

# Create 2 OCHRE battery instances
print(f'Creating OCHRE (v{__version__}) Battery models')
b1 = Battery(**battery_properties)
b2 = Battery(**battery_properties2)
battery_dict = {
    'b1': b1,
    'b2': b2,
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

    return {
        'State of Charge (-)': soc_new,
        'Energy Capacity (kWh)': capacity_kwh,
        'Power Capacity (kW)': battery.capacity,
        # calculating "average" efficiency at (max power capacity / 2)
        'Charge Efficiency (-)': battery.calculate_efficiency(battery.capacity / 2),
        'Discharge Efficiency (-)': battery.calculate_efficiency(- battery.capacity / 2),
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

# Run OCHRE model for 1 time step (no controls for first time step)
for name, battery in battery_dict.items():
    controls = {}
    battery.update(1, temperature_schedule, controls)
    battery.update_model(None)
    power_to_dss = battery.electric_kw  # in kW, power output of battery to send to grid simulator

# Get updated parameters from OCHRE models
models = {name: get_ochre_properties(battery) for name, battery in battery_dict.items()}

# Send parameters to aggregator
agg.update_parameters(**models)

# Aggregate model to create virtual battery model
virtual_model = agg.aggregate()
print('Virtual battery model parameters:', virtual_model)

# Send virtual model to FRS and receive virtual battery setpoint
virtual_setpoint = 10  # in kW

# Dispatch setpoint to individual batteries
setpoints = agg.dispatch(p_setpoint=virtual_setpoint)
print(f'Setpoints for individual batteries with a virtual battery setpoint of {virtual_setpoint}: {setpoints}')

# ********* Next Time Step ************

# At next temp step, run OCHRE with battery setpoint controls
for name, battery in battery_dict.items():
    controls = {'P Setpoint': setpoints.get(name)}
    battery.update(1, temperature_schedule, controls)
    battery.update_model(None)
    power_to_dss = battery.electric_kw  # in kW, power output of battery to send to grid simulator
