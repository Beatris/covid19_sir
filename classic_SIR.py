import numpy as np
import pandas as pd
import lmfit
import matplotlib.pyplot as plt

from typing import Union

from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

from data import get_SIR_data


class SIRData:

    def __init__(self, times, S, I, R, N, beta, gamma):
        self.S = S
        self.I = I
        self.R = R
        self.N = N
        self.t = times
        self.beta = beta
        self.gamma = gamma

    @property
    def _params_text(self) -> str:
        return '\n'.join([
            r'$\beta$ = {:.3f}'.format(self.beta),
            r'$\gamma$ = {:.3f}'.format(self.gamma),
            r'$R0 = {:.3f}$'.format(self.beta / self.gamma)
        ])

    def subplot(self, ax, S=True, I=True, R=True, text_position=(0.6, 0.5), title=None):
        if S:
            ax.plot(self.t, self.S, 'b', alpha=0.5, lw=2, label='Susceptible')
        if I:
            ax.plot(self.t, self.I, 'r', alpha=0.5, lw=2, label='Infectious')
        if R:
            ax.plot(self.t, self.R, 'g', alpha=0.5, lw=2, label='Recovered')
        ax.set_xlabel('Time (in days)')
        ax.set_ylabel('Population size')
        ax.text(
            *text_position, self._params_text,
            horizontalalignment='left',
            verticalalignment='center',
            transform = ax.transAxes
        )
        if title:
            ax.set_title(title)

    def plot(self, num=None, scale='linear', title=None):
        plt.figure(num=num, dpi=100)
#         plt.plot(self.t, self.S, 'b', alpha=0.5, lw=2, label='Susceptible')
        plt.plot(self.t, self.I, 'r', alpha=0.5, lw=2, label='Infectious')
#         plt.plot(self.t, self.R, 'g', alpha=0.5, lw=2, label='Recovered')
        plt.xlabel('Time (in days)')
        plt.ylabel('Population size')
        plt.yscale(scale)
        plt.figtext(.67, .74, self._params_text)
        if title:
            plt.title(title)
        plt.legend()
        plt.show()


class SIRModel:

    def __init__(self, initial_conditions: tuple, **params):
        self.initial_conditions = initial_conditions

    @classmethod
    def _get_data_type(cls):
        return SIRData

    @staticmethod
    def _ode(t, y, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def _solve(self, times: np.ndarray, *args, initial_cond=None):
        initial_cond = initial_cond if initial_cond else self.initial_conditions
        solution = solve_ivp(
            self._ode, (times[0], times[-1]), initial_cond,
            args=args, dense_output=True
        )
        return solution.sol(times)

    def solve(self, times: np.ndarray, *args, initial_cond=None):
        result = self._solve(times, *args, initial_cond=initial_cond)
        wrapper = self._get_data_type()
        return wrapper(times, *result, *args)

    def fit(
            self, S_obs, I_obs, R_obs,
            initial_guess=None,
            fit_S=True, fit_I=True, fit_R=True,
            extrapolate_days=0
        ):
        times = np.array(range(len(I_obs)))
        N = S_obs[0] + I_obs[0] + R_obs[0]
        observed_data = []
        if fit_S:
            observed_data.append(S_obs)
        if fit_I:
            observed_data.append(I_obs)
        if fit_R:
            observed_data.append(R_obs)
        observed_data = np.array(observed_data).flatten()

        def target(t, beta, gamma):
            solution = self._solve(t, N, beta, gamma)
            values = []
            if fit_S:
                values.append(solution[0])
            if fit_I:
                values.append(solution[1])
            if fit_R:
                values.append(solution[2])
            return np.array(values).flatten()

        model = lmfit.Model(target)
        if initial_guess:
            for param_name, value_kwargs in initial_guess.items():
                model.set_param_hint(param_name, min=0, **value_kwargs)
        params = model.make_params()
        result = model.fit(observed_data, params, method="leastsq", t=times)  # fitting
        beta = result.best_values['beta']
        gamma = result.best_values['gamma']
        times = np.append(times, [np.array(range(len(I_obs), len(I_obs)+extrapolate_days))])
        return self.solve(times, N, beta, gamma)


class SIRDataLogisticR0(SIRData):
    
    def __init__(self, times, S, I, R, N, gamma, R0_start, R0_end, x0, k):
        super().__init__(times, S, I, R, N, None, gamma)

    @property
    def _params_text(self) -> str:
        return ''


class SIRModelLogisticR0(SIRModel):
    
    @classmethod
    def _get_data_type(cls):
        return SIRDataLogisticR0

    @staticmethod
    def _ode(t, y, N, gamma, R0_start, R0_end, x0, k):

        def logistic_R_0(t):
            return (R0_start-R0_end) / (1 + np.exp(-k*(-t+x0))) + R0_end

        def beta(t):
            return logistic_R_0(t) * gamma

        S, I, R = y
        dSdt = -beta(t) * S * I / N
        dIdt = beta(t) * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def fit(
            self, S_obs, I_obs, R_obs,
            initial_guess=None,
            fit_S=True, fit_I=True, fit_R=True,
            extrapolate_days=0
        ):
        times = np.array(range(len(I_obs)))
        N = S_obs[0] + I_obs[0] + R_obs[0]
        observed_data = []
        if fit_S:
            observed_data.append(S_obs)
        if fit_I:
            observed_data.append(I_obs)
        if fit_R:
            observed_data.append(R_obs)
        observed_data = np.array(observed_data).flatten()

        def target(t, gamma, R0_start, R0_end, x0, k):
            solution = self._solve(
                t, N, gamma, R0_start, R0_end, x0, k,
            )
            values = []
            if fit_S:
                values.append(solution[0])
            if fit_I:
                values.append(solution[1])
            if fit_R:
                values.append(solution[2])
            return np.array(values).flatten()

        model = lmfit.Model(target)
        if initial_guess:
            for param_name, value_kwargs in initial_guess.items():
                model.set_param_hint(param_name, **value_kwargs)
        params = model.make_params()
        result = model.fit(observed_data, params, method="leastsq", t=times)  # fitting
        popt = result.best_values
        print(popt)
        times = np.append(times, [np.array(range(len(I_obs), len(I_obs)+extrapolate_days))])
        return self.solve(
            times, N, popt['gamma'],
            popt['R0_start'], popt['R0_end'], popt['x0'], popt['k']
        )


def fit_sir(
        country_name: str, population_size: int,
        model=SIRModel, initial_guess=None, plot_R=True,
        fit_I=True, fit_R=True, fit_S=True, extrapolate_days=0
    ):
    data = get_SIR_data(country_name, population_size)
    S, I, R = data['S'], data['I'], data['R']

#     if outbreak_shift > 0: # maybe makes sense only when fitting dR/dt only
#         S = np.concatenate((np.array([S[0]-1]*outbreak_shift), S))
#         I = np.concatenate((np.array([1]*outbreak_shift), I))
#         R = np.concatenate((np.zeros(outbreak_shift), R))

    fig, (ax_l, ax_r) = plt.subplots(nrows=1, ncols=2, dpi=100, figsize=(12,4))
    # plot data (left - linear scale):
    ax_l.plot(np.array(range(len(I))), I, 'o', alpha=0.5, lw=2, label='Infectious (observed)')
    # plot data (right - log scale)
    ax_r.set_yscale('log')
    ax_r.plot(np.array(range(len(I))), I, 'o', alpha=0.5, lw=2, label='Infectious (observed)')

    fitted_data = model((S[0], I[0], R[0])).fit(
        S, I, R, initial_guess=initial_guess,
        fit_S=fit_S, fit_I=fit_I, fit_R=fit_R, extrapolate_days=extrapolate_days
    )
    # plot fit (left - linear scale):
    fitted_data.subplot(ax_l, S=False, R=False, text_position=(0.1, 0.8), title='Linear scale')
    # plot fit (right - linear scale):
    fitted_data.subplot(ax_r, S=False, R=False, text_position=(0.1, 0.8), title='Log scale')
    
    if plot_R:
        fig2, (ax2_l, ax2_r) = plt.subplots(nrows=1, ncols=2, dpi=100, figsize=(12,4))
        ax2_l.plot(np.array(range(len(R))), R, 'o', alpha=0.5, lw=2, label='Recovered (observed)')
        ax2_r.set_yscale('log')
        ax2_r.plot(np.array(range(len(R))), R, 'o', alpha=0.5, lw=2, label='Recovered (observed)')
        # plot fit (left - linear scale):
        fitted_data.subplot(ax2_l, S=False, I=False, text_position=(0.1, 0.8), title='Linear scale')
        # plot fit (right - linear scale):
        fitted_data.subplot(ax2_r, S=False, I=False, text_position=(0.1, 0.8), title='Log scale')
        fig2.suptitle(country_name + ' Recovered')

    fig.suptitle(country_name + ' Infectious')
    plt.show()
