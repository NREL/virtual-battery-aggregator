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

# Aggregate models (all)
virtual_model = agg.aggregate()
print('Aggregated model parameters:', virtual_model)

# Aggregate models (selection)
virtual_model = agg.aggregate(['A', 'B', 'C'])
print('Aggregated model parameters (from selection):', virtual_model)

# Dispatch power setpoint
setpoints = agg.dispatch(p_setpoint=1)
print('Dispatch setpoints for 1 kW (charge) virtual setpoint:', setpoints)

setpoints = agg.dispatch(p_setpoint=-1)
print('Dispatch setpoints for -1 kW (discharge) virtual setpoint:', setpoints)

setpoints = agg.dispatch(p_setpoint=10)
print('Dispatch setpoints for 10 kW (charge) virtual setpoint:', setpoints)

setpoints = agg.dispatch(p_setpoint=-8)
print('Dispatch setpoints for -8 kW (discharge) virtual setpoint:', setpoints)

# Update and redispatch
parameter_updates = {
    'A': {'State of Charge (-)': 0.01},
    'B': {'State of Charge (-)': 0},
}
agg.update_parameters(**parameter_updates)
setpoints = agg.dispatch(p_setpoint=-8)
print('Updated dispatch setpoints for -8 kW (discharge) virtual setpoint:', setpoints)
