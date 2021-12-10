import numpy as np
import pandas as pd
import cvxpy as cp

pd.set_option('expand_frame_repr', False)  # Keeps results on 1 line

# Convert Aggregator to combine OCHRE Battery models into a virtual battery model

# Required parameters for virtual battery model, and their min/max values
BATTERY_PARAMETERS = {
    'State of Charge (-)': (0, 1),  # assumed min and max of 0-100%
    'Energy Capacity (kWh)': (0, np.inf),
    'Power Capacity (kW)': (0, np.inf),  # for both charge and discharge
    'Charge Efficiency (-)': (0, 1),
    'Discharge Efficiency (-)': (0, 1),
}


def weighted_average(values, weights):
    return (values * weights).sum() / weights.sum()


class BatteryAggregator:

    def __init__(self, time_res, models=None):
        self.time_res = time_res
        self.models = pd.DataFrame(columns=list(BATTERY_PARAMETERS.keys()))

        # parameters for aggregation
        self.aggregated_model = None
        self.aggregated_model_names = None
        self.charge_max_power = None  # Max charge power in single time step, per battery, in kW
        self.discharge_max_power = None  # Max discharge power in single time step, per battery, in kW

        # optimization for disaggregation
        self.parameters = None
        self.problem = None

        if models is not None:
            self.add_models(**models)
            self.aggregate()
            self.make_opt_problem()

    def add_models(self, **models):
        # takes a dictionary of battery models of the form {name: {parameter_name: parameter_value}}
        # Required parameters:
        #  - 'State of Charge (-)': Battery SOC, fraction from 0 to 1
        #  - 'Energy Capacity (kWh)': Battery energy capacity associated with SOC=1
        #  - 'Power Capacity (kW)': Maximum battery power
        #  - 'Charge Efficiency (-)': Efficiency of charging
        #  - 'Discharge Efficiency (-)': Efficiency of discharging
        df_models = pd.DataFrame(models).T
        df_models.index.name = 'Model'

        # Check for missing/bad parameters
        missing = [name for name in BATTERY_PARAMETERS if name not in df_models.columns]
        bad = [name for name in df_models.columns if name not in BATTERY_PARAMETERS]
        if missing or bad:
            raise Exception(f'Error parsing battery parameters. '
                            f'Invalid parameters: {bad}; Missing parameters: {missing}')

        # Check for missing/bad values
        param_min = pd.Series({name: val[0] for name, val in BATTERY_PARAMETERS.items()})
        param_max = pd.Series({name: val[1] for name, val in BATTERY_PARAMETERS.items()})
        bad_values = df_models.isna() | (df_models < param_min) | (df_models > param_max)
        if bad_values.any().any():
            bad_values = bad_values.reset_index().melt('Model', var_name='Parameter')
            df_models = df_models.reset_index().melt('Model', var_name='Parameter').loc[bad_values['value']]
            raise Exception(f'Error parsing battery parameter values: {df_models.iloc[:5]}')

        self.models = self.models.append(df_models, sort=False)

    def make_opt_problem(self):
        # set up an optimization problem to dispatch a virtual battery power setpoint to individual batteries\
        # objective is to minimize the maximum power of an individual battery, i.e. min max(p_i / p_max)
        # constraints on battery SOC, max power, and total power setpoint
        n = len(self.models)

        # Create variables and parameters
        self.parameters = {
            'p_chg': cp.Variable((n,), name='p_chg', nonneg=True),
            'p_dis': cp.Variable((n,), name='p_dis', nonneg=True),
            'p_set': cp.Parameter(name='p_set'),
            'p_max': cp.Parameter((n,), name='p_max', nonneg=True),
            'p_max_inv': cp.Parameter((n,), name='p_max_inv', nonneg=True),
            'soc': cp.Parameter((n,), name='soc', nonneg=True),
            'eta_chg': cp.Parameter((n,), name='eta_chg', nonneg=True),
            'eta_dis': cp.Parameter((n,), name='eta_dis', nonneg=True),
        }

        # Create objective: minimize max(abs(p_i) / p_i_max)
        # Add small cost for charging and discharging to eliminate simultaneous charge/discharge
        # see https://arxiv.org/abs/1805.00100
        p_chg = self.parameters['p_chg']
        p_dis = self.parameters['p_dis']
        objective = cp.Minimize(
            cp.maximum(cp.max(cp.multiply(p_chg, self.parameters['p_max_inv'])),
                       cp.max(cp.multiply(p_dis, self.parameters['p_max_inv'])))
            + cp.sum(cp.multiply(p_chg + p_dis, self.parameters['p_max_inv']))
        )

        # Create constraints: SOC constraints, setpoint constraint
        soc_new = self.parameters['soc'] + (cp.multiply(p_chg, self.parameters['eta_chg']) -
                                            cp.multiply(p_dis, self.parameters['eta_dis']))
        constraints = [
            soc_new >= 0,
            soc_new <= 1,
            p_chg <= self.parameters['p_max'],
            p_dis <= self.parameters['p_max'],
            cp.sum(p_chg - p_dis) == self.parameters['p_set']
        ]

        # Create optimization problem
        self.problem = cp.Problem(objective, constraints)

        # check for DCP
        if not self.problem.is_dcp():
            if not objective.is_dcp():
                raise Exception('Cost function is not convex.')
            for c in constraints:
                if not c.is_dcp():
                    raise Exception('Constraint is not valid for DCP: {}'.format(c))

        if not self.problem.is_dpp():
            if not objective.is_dpp():
                raise Exception('Cost function is not DPP.')
            for c in constraints:
                if not c.is_dpp():
                    raise Exception('Constraint is not DPP: {}'.format(c))

    def update_models(self, **models):
        # Update individual models and aggregated model
        if models:
            df_updates = pd.DataFrame(models).T
            self.models.update(df_updates)
            self.aggregate()

        # Update optimization parameters
        self.parameters['p_max'].value = self.models['Power Capacity (kW)'].values
        self.parameters['p_max_inv'].value = 1 / self.models['Power Capacity (kW)'].values
        self.parameters['soc'].value = self.models['State of Charge (-)'].values

        # eta accounts for efficiency, energy capacity, and time resolution. Discharge efficiency is inverted
        kw_capacity = self.models['Energy Capacity (kWh)'].values / self.time_res.total_seconds() * 3600
        self.parameters['eta_chg'].value = self.models['Charge Efficiency (-)'].values / kw_capacity
        self.parameters['eta_dis'].value = 1 / self.models['Discharge Efficiency (-)'].values / kw_capacity

    def aggregate(self, model_names=None):
        # combine models into single, virtual model
        # model_names is a list of model names to combine. By default, combines all models
        # returns dictionary of virtual battery model
        self.aggregated_model_names = model_names if model_names is not None else self.models.index
        df = self.models.loc[self.aggregated_model_names, :]

        # determine max charge and discharge power per battery
        soc_kwh = df['State of Charge (-)'] * df['Energy Capacity (kWh)']
        hours = self.time_res.total_seconds() / 3600
        full_charge_power = (df['Energy Capacity (kWh)'] - soc_kwh) / hours / df['Charge Efficiency (-)']
        self.charge_max_power = full_charge_power.clip(lower=0, upper=df['Power Capacity (kW)'])
        full_discharge_power = soc_kwh / hours * df['Discharge Efficiency (-)']
        self.discharge_max_power = full_discharge_power.clip(lower=0, upper=df['Power Capacity (kW)'])

        # SOC is weighted average of energy capacity
        soc = weighted_average(df['State of Charge (-)'], df['Energy Capacity (kWh)'])

        # Efficiency is weighted average of max power
        eff_charge = weighted_average(df['Charge Efficiency (-)'], self.charge_max_power)
        eff_discharge = weighted_average(df['Discharge Efficiency (-)'], self.discharge_max_power)

        self.aggregated_model = {
            'State of Charge (-)': soc,
            'Energy Capacity (kWh)': df['Energy Capacity (kWh)'].sum(),
            'Power Capacity (kW)': df['Power Capacity (kW)'].sum(),
            'Charge Efficiency (-)': eff_charge,
            'Discharge Efficiency (-)': eff_discharge,
        }
        return self.aggregated_model

    def dispatch(self, p_setpoint, fail_on_error=True, model_names=None):
        # Dispatch a power setpoint (in kW) for the virtual battery
        # See self.make_opt_problem for details on objective and constraints
        # Returns a dictionary of {name: setpoint} pairs
        if model_names is not None and model_names != self.aggregated_model_names:
            # redo aggregation if list of models has changed
            self.aggregate(model_names)

        # Force setpoint within achievable limits
        if not (-self.discharge_max_power.sum() <= p_setpoint <= self.charge_max_power.sum()):
            new_setpoint = min(max(p_setpoint, -self.discharge_max_power.sum()), self.charge_max_power.sum())
            print(f'WARNING: Virtual power setpoint of {p_setpoint:.3f} kW is infeasible.'
                  f' Closest achievable setpoint is {new_setpoint:.3f} kW.')
            p_setpoint = new_setpoint
        self.parameters['p_set'].value = p_setpoint

        # solve optimization problem
        try:
            opt_value = self.problem.solve()
        except cp.error.SolverError as e:
            opt_value = None
            print('Solver error:', e)

        p_chg = self.parameters['p_chg'].value
        p_dis = self.parameters['p_dis'].value
        if 'infeasible' in self.problem.status or 'unbounded' in self.problem.status:
            # If problem didn't solve, fail or raise a warning
            if fail_on_error:
                raise Exception(f'Optimization failed with status {self.problem.status} and value {opt_value}.')
            else:
                print(f'WARNING: Optimization failed with status {self.problem.status} and value {opt_value}.')
                setpoints = np.zeros(len(self.models))
        elif not all([min(abs(c), abs(d)) < 0.01 for c, d in zip(p_chg, p_dis)]):
            if fail_on_error:
                raise Exception(f'Simultaneous charging and discharging issue. Optimization values: {self.parameters}')
            else:
                print(f'Simultaneous charging and discharging issue. Optimization values: {self.parameters}')
                setpoints = np.zeros(len(self.models))
        else:
            setpoints = p_chg - p_dis

        # Return dictionary of setpoints
        return dict(zip(self.models.index, setpoints))
