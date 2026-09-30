from iminuit import Minuit
import numpy as np
from scipy.signal import fftconvolve

def fit_function(data, bins, function, initial_guess, limits=None):
    """
    Fit a function to a histogram of data.

    Arguments:
    - data (array-like): The data to fit.
    - bins (int or array-like): The number of bins or the bin edges.
    - function (callable): The function to fit.
    - initial_guess (array-like): The initial guess for the function parameters.

    Keywords:
    - limits (array-like): The limits for the function parameters. Default is None, which means no limits are applied.

    Returns:
    - tuple: A tuple containing the fitted parameters, their errors, and the chi-squared value.
    """
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = np.diff(bin_edges)
    sample_size = counts.sum()
    hist = counts / (sample_size * bin_widths)
    hist_errors = np.sqrt(counts) / (sample_size * bin_widths)
    fit_mask = counts > 0
    
    def _chi2(*parameters):
        """
        Calculate the chi-squared value for the fit.

        Arguments:
        - parameters (array-like): The parameters for the function.

        Returns:
        - float: The chi-squared value.
        """
        expected = np.asarray(
            function(bin_centers, *parameters),
            dtype=float,
        )

        valid = (
            fit_mask
            & np.isfinite(expected)
            & (expected > 0)
        )
        if not np.all(valid[fit_mask]):
            return 1e100

        residuals = (hist[valid] - expected[valid]) / hist_errors[valid]
        return np.sum(residuals ** 2)

    parameter_names = [
        f"parameter_{index}"
        for index in range(len(initial_guess))
    ]

    minimizer = Minuit(
        _chi2,
        *initial_guess,
        name=parameter_names,
    )
    if limits is not None:
        minimizer.limits = limits
        
    minimizer.errordef = Minuit.LEAST_SQUARES
    minimizer.migrad(ncall=10000)

    if minimizer.fmin.is_above_max_edm or minimizer.fmin.has_reached_call_limit:
        raise RuntimeError(
            f"Minuit failed to find a minimum: {minimizer.fmin}"
        )

    return np.asarray(minimizer.values), np.asarray(minimizer.errors), minimizer.fval

def gaussian(x, mean, stddev):
    """
    Normalized Gaussian distribution.

    Arguments:
    - x (array-like): The input values.
    - mean (float): The mean of the Gaussian distribution.
    - stddev (float): The standard deviation of the Gaussian distribution.

    Returns:
    - array-like: The values of the Gaussian distribution at the input values.
    """
    return 1 / (np.sqrt(2 * np.pi) * stddev) * np.exp(
        -((x - mean) ** 2) / (2 * stddev ** 2)
    )


def crystal_ball_left(x, mean, stddev, alpha, n):
    """
    Crystal Ball shape with a power-law tail on the low-x side.

    Arguments:
    - x (array-like): The input values.
    - mean (float): The mean of the Gaussian core.
    - stddev (float): The standard deviation of the Gaussian core.
    - alpha (float): The point where the power-law tail begins.
    - n (float): The exponent of the power-law tail.

    Returns:
    - array-like: The values of the Crystal Ball function at the input values.
    """
    z = (x - mean) / stddev
    alpha = abs(alpha)

    A = (n / alpha) ** n * np.exp(-alpha ** 2 / 2)
    B = n / alpha - alpha
    tail_base = np.maximum(B - z, np.finfo(float).eps)

    shape = np.empty_like(z, dtype=float)
    gaussian_mask = z > -alpha
    shape[gaussian_mask] = np.exp(-z[gaussian_mask] ** 2 / 2)
    shape[~gaussian_mask] = A * tail_base[~gaussian_mask] ** (-n)

    return shape / (np.sqrt(2 * np.pi) * stddev)


def breit_wigner(x, mass, width):
    """
    Normalized non-relativistic Breit-Wigner distribution.

    Arguments:
    - x (array-like): The input values.
    - mass (float): The mass parameter of the Breit-Wigner distribution.
    - width (float): The width parameter of the Breit-Wigner distribution.

    Returns:
    - array-like: The values of the Breit-Wigner distribution at the input values.
    """
    half_width = width / 2
    return 1 / np.pi * half_width / (
        (x - mass) ** 2 + half_width ** 2
    )

def exponential_decay(x, amplitude, decay_constant):
    """
    Exponential decay function.

    Arguments:
    - x (array-like): The input values.
    - amplitude (float): The amplitude of the exponential decay.
    - decay_constant (float): The decay constant of the exponential decay.

    Returns:
    - array-like: The values of the exponential decay function at the input values.
    """
    return amplitude * np.exp(-decay_constant * x)

def breit_wigner_crystal_ball(
    x, amplitude, mean_bw, width_bw, sigma_cb, alpha_cb, n_cb
):
    """
    Breit-Wigner convolved with a low-mass Crystal Ball response.

    Arguments:
    - x (array-like): The input values.
    - amplitude (float): The amplitude of the Breit-Wigner distribution.
    - mean_bw (float): The mean of the Breit-Wigner distribution.
    - width_bw (float): The width of the Breit-Wigner distribution.
    - sigma_cb (float): The standard deviation of the Crystal Ball distribution.
    - alpha_cb (float): The alpha parameter of the Crystal Ball distribution.
    - n_cb (float): The n parameter of the Crystal Ball distribution.

    Returns:
    - array-like: The values of the convolved distribution at the input values.
    """
    x = np.asarray(x, dtype=float)
    step = 0.02
    bw_grid = np.arange(40.0, 151.0 + step, step)
    response_grid = np.arange(-110.0, 111.0 + step, step)

    bw_values = breit_wigner(bw_grid, mean_bw, width_bw)
    response_values = crystal_ball_left(
        response_grid, 0.0, sigma_cb, alpha_cb, n_cb
    )
    response_values /= np.trapezoid(response_values, response_grid)

    convolution = fftconvolve(bw_values, response_values, mode="full") * step
    convolution_grid = (
        bw_grid[0]
        + response_grid[0]
        + np.arange(convolution.size) * step
    )

    result = np.interp(x, convolution_grid, convolution, left=0.0, right=0.0)
    return amplitude * result


def crystal_ball_double(
    x, mean, stddev, alpha_left, n_left, alpha_right, n_right
):
    """
    Crystal Ball shape with independent low- and high-x power-law tails.
    
    Arguments:
    - x (array-like): The input values.
    - mean (float): The mean of the Crystal Ball distribution.
    - stddev (float): The standard deviation of the Crystal Ball distribution.
    - alpha_left (float): The alpha parameter of the left tail.
    - n_left (float): The n parameter of the left tail.
    - alpha_right (float): The alpha parameter of the right tail.
    - n_right (float): The n parameter of the right tail.

    Returns:
    - array-like: The values of the Crystal Ball distribution at the input values.
    """
    z = (x - mean) / stddev
    alpha_left = abs(alpha_left)
    alpha_right = abs(alpha_right)

    A_left = (n_left / alpha_left) ** n_left * np.exp(-alpha_left ** 2 / 2)
    B_left = n_left / alpha_left - alpha_left
    A_right = (n_right / alpha_right) ** n_right * np.exp(-alpha_right ** 2 / 2)
    B_right = n_right / alpha_right - alpha_right

    shape = np.empty_like(z, dtype=float)
    left_mask = z <= -alpha_left
    right_mask = z >= alpha_right
    core_mask = ~(left_mask | right_mask)

    shape[left_mask] = A_left * np.maximum(
        B_left - z[left_mask], np.finfo(float).eps
    ) ** (-n_left)
    shape[core_mask] = np.exp(-z[core_mask] ** 2 / 2)
    shape[right_mask] = A_right * np.maximum(
        B_right + z[right_mask], np.finfo(float).eps
    ) ** (-n_right)

    return shape / (np.sqrt(2 * np.pi) * stddev)


def data_fit_ratio(x, ratio):
    """
    Function that combines a Breit-Wigner convolved with a Crystal Ball function and an exponential decay function, weighted by a given ratio.

    Arguments:
    - x (array-like): The input values.
    - ratio (float): The ratio of the two components.

    Returns:
    - array-like: The values of the fitted distribution at the input values.
    """
    return (
        ratio * breit_wigner_crystal_ball(x, *vals_egam1)
        + (1 - ratio) * exponential_decay(x, *vals_egam7)
    )

def true_data_fit(x, amplitude_sig, amplitude_bkg, mean_bw, width_bw, sigma_cb, alpha_cb, n_cb, decay_constant):
    """
    Function that combines a Breit-Wigner convolved with a Crystal Ball function and an exponential decay function, weighted by their respective amplitudes.

    Arguments:
    - x (array-like): The input values.
    - amplitude_sig (float): The amplitude of the signal component.
    - amplitude_bkg (float): The amplitude of the background component.
    - mean_bw (float): The mean of the Breit-Wigner distribution.
    - width_bw (float): The width of the Breit-Wigner distribution.
    - sigma_cb (float): The sigma parameter of the Crystal Ball distribution.
    - alpha_cb (float): The alpha parameter of the Crystal Ball distribution.
    - n_cb (float): The n parameter of the Crystal Ball distribution.
    - decay_constant (float): The decay constant of the exponential decay function.

    Returns:
    - array-like: The values of the fitted distribution at the input values.
    """
    return (
        breit_wigner_crystal_ball(x, amplitude_sig, mean_bw, width_bw, sigma_cb, alpha_cb, n_cb)
        + exponential_decay(x, amplitude_bkg, decay_constant)
    )