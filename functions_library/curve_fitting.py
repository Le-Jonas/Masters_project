from iminuit import Minuit
import numpy as np

def fit_function(data, bins, function, initial_guess, limits=None):
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = np.diff(bin_edges)
    sample_size = counts.sum()
    hist = counts / (sample_size * bin_widths)
    hist_errors = np.sqrt(counts) / (sample_size * bin_widths)
    fit_mask = counts > 0

    def chi2(*parameters):
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
        chi2,
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

    return np.asarray(minimizer.values)

def gaussian(x, amplitude, mean, stddev):
    return amplitude / (np.sqrt(2 * np.pi) * stddev) * np.exp(
        -((x - mean) ** 2) / (2 * stddev ** 2)
    )


def crystal_ball_left(x, amplitude, mean, stddev, alpha, n):
    """Crystal Ball shape with a power-law tail on the low-x side."""
    z = (x - mean) / stddev
    alpha = abs(alpha)

    A = (n / alpha) ** n * np.exp(-alpha ** 2 / 2)
    B = n / alpha - alpha
    tail_base = np.maximum(B - z, np.finfo(float).eps)

    shape = np.empty_like(z, dtype=float)
    gaussian_mask = z > -alpha
    shape[gaussian_mask] = np.exp(-z[gaussian_mask] ** 2 / 2)
    shape[~gaussian_mask] = A * tail_base[~gaussian_mask] ** (-n)

    return amplitude * shape / (np.sqrt(2 * np.pi) * stddev)


def crystal_ball_double(
    x, amplitude, mean, stddev, alpha_left, n_left, alpha_right, n_right
):
    """Crystal Ball shape with independent low- and high-x power-law tails."""
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

    return amplitude * shape / (np.sqrt(2 * np.pi) * stddev)

def exponential_decay(x, amplitude, decay_constant):
    return amplitude * np.exp(-decay_constant * x)

def data_fit_func(x, ratio):
    return (
        ratio * crystal_ball_double(x, *vals_egam1)
        + (1 - ratio) * exponential_decay(x, *vals_egam7)
    )