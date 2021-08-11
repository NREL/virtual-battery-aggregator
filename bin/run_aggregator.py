import datetime as dt

from aggregator import BatteryAggregator, BATTERY_PARAMETERS

# Batter model parameters and other parameters
time_res = dt.timedelta(minutes=30)
models = {
    'A': (0.5, 10, 5, 0.99, 0.99),
    'B': (0.1, 10, 5, 0.98, 0.99),
    'C': (0.9, 5, 3, 0.98, 0.99),
    'D': (0.5, 5, 3, 0.99, 0.98),
    'E': (0.1, 2, 2, 0.99, 0.99),
    'F': (0.9, 2, 2, 0.98, 0.98),
}
models = {name: {param_name: val for param_name, val in zip(BATTERY_PARAMETERS, vals)} for name, vals in models.items()}

model_a = models['A']
print(f'Battery Model A: {model_a}')

# Create Battery Aggregator
agg = BatteryAggregator(time_res, models=models)

# Alternate method:
# agg = BatteryAggregator(time_res)
# for name, model in models.items():
#     agg.add_models(**{name: model})
# agg.make_opt_problem()

# Show model parameters
print('Individual model parameters:')
print(agg.models)
print()


def round_dict(d, digits=3):
    return {key: round(val, digits) for key, val in d.items()}


# Aggregate models (all)
virtual_model = agg.aggregate()
print('Aggregated model parameters:', round_dict(virtual_model))

# Aggregate models (selection)
virtual_model = agg.aggregate(['A', 'B', 'C'])
print('Aggregated model parameters (from selection):', round_dict(virtual_model))
print()

# Dispatch power setpoint
setpoints = agg.dispatch(p_setpoint=1)
print('Dispatch setpoints for 1 kW (charge) virtual setpoint:', round_dict(setpoints))

setpoints = agg.dispatch(p_setpoint=10)
print('Dispatch setpoints for 10 kW (charge) virtual setpoint:', round_dict(setpoints))

setpoints = agg.dispatch(p_setpoint=-8)
print('Dispatch setpoints for -8 kW (discharge) virtual setpoint:', round_dict(setpoints))

print('Running dispatch with an *infeasible* virtual setpoint of 25 kW (discharge)...')
setpoints = agg.dispatch(p_setpoint=25)
print('Dispatch setpoints for 25 kW (charge) virtual setpoint:', round_dict(setpoints))
print()

# Update and redispatch
parameter_updates = {
    'A': {'State of Charge (-)': 0.1},
    'B': {'State of Charge (-)': 0.05},
    'C': {'State of Charge (-)': 0.1},
    'D': {'State of Charge (-)': 0},
    'E': {'State of Charge (-)': 0},
    'F': {'State of Charge (-)': 0},
}
agg.update_parameters(**parameter_updates)
print('Updating model parameters with low SOC:')
print(agg.models)
print()

setpoints = agg.dispatch(p_setpoint=-1)
print('Updated dispatch setpoints for -1 kW (discharge) virtual setpoint:', round_dict(setpoints))

print('Running dispatch with an *infeasible* virtual setpoint of -6 kW (discharge)...')
setpoints = agg.dispatch(p_setpoint=-6)
print('Updated dispatch setpoints for -6 kW (discharge) virtual setpoint:', round_dict(setpoints))
