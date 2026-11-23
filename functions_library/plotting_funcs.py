import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from pathlib import Path

from .curve_fitting import fit_function, fit_data_to_mc, true_data_fit

def _get_truth_totals(predictions : ArrayLike, truth : ArrayLike, origin_atlas : ArrayLike, z_masses_truth : ArrayLike | None = None, n_total : int = 200, bins : int | list[float] = np.arange(54.95, 135.05, 0.1),) -> tuple[np.ndarray[int], np.ndarray[int], int, int]:
    """
    Given a model, a full data loader, and a target data loader, compute the total number of true positives and true negatives for different prediction thresholds and the cut true positives and true negatives. This function is useful for evaluating the performance of a binary classification model.

    Arguments:
    - model: A PyTorch model used for making predictions.
    - full_loader: A DataLoader that provides the full dataset for making predictions.
    - target_loader: A DataLoader that provides the target labels for computing true positives and true negatives.
    - device: The device (CPU or GPU) on which the model and data are located.
    - dataloader_kwargs: A dictionary of keyword arguments to pass to the DataLoader constructor.
    - valid_fields: A dictionary of needed non_nan fields for the dataset.

    Keywords:
    - n_total: The number of thresholds to evaluate. Default is 200.
    - sample_size: The number of samples to use for computing the mean and standard deviation. Default is 100,000.

    Returns:
    - N_tp: An array containing the number of true positives for each threshold.
    - N_fp: An array containing the number of false positives for each threshold.
    - N_tp_cut: The number of true positives for the cut threshold.
    - N_fp_cut: The number of false positives for the cut threshold.
    """
    if bins is not None and z_masses_truth is not None:
        mass_in_range = (z_masses_truth / 1000 > bins[0]) & (z_masses_truth / 1000 < bins[-1])
        truth = truth[mass_in_range]
        origin_atlas = origin_atlas[mass_in_range]
        predictions = predictions[mass_in_range]

    N_tp_cut = (truth * origin_atlas).sum()
    N_fp_cut = ((1 - truth) * origin_atlas).sum()
        
    N_tp = np.zeros(n_total)
    N_fp = np.zeros(n_total)
    prediction_values = np.linspace(-10, 10, n_total)
    for i in range(n_total):
        predicted_labels = (predictions > prediction_values[i]).astype(bool)
        N_tp[i] = (truth * predicted_labels).sum()
        N_fp[i] = ((1 - truth) * predicted_labels).sum()
    return N_tp, N_fp, N_tp_cut, N_fp_cut

def _get_totals_from_distribution(
    predictions : ArrayLike,
    data_masses : ArrayLike,
    mc_signal_hist : ArrayLike | None = None, 
    mc_background_hist : ArrayLike | None = None,
    bins : int | list[float] = np.arange(54.95, 135.05, 0.1),
    n_total : int = 200,
    plotting_path : str | Path | None = None,
    save_namespace : str | None = None,
    data_truth : ArrayLike | None = None,
    mc_signal_limit : int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Finds the total number of true positives and true negatives for different prediction thresholds by fitting the data to the Monte Carlo signal and background histograms.

    Arguments:
    - files_path: A list of paths to the H5 files containing the data.
    - model: A PyTorch model used for making predictions.
    - device: The device (CPU or GPU) on which the model and data are located
    - mc_signal_hist: The Monte Carlo signal histogram.
    - mc_background_hist: The Monte Carlo background histogram.
    - data_masses: The Z boson masses from the data.
    - dataloader_kwargs: A dictionary of keyword arguments to pass to the DataLoader constructor
    - valid_fields: A dictionary of needed non_nan fields for the dataset.
    - use_fit_template: A boolean indicating whether to use template fit. Default is False.
    - predictions_path: A path to a file containing pre-computed predictions. If provided, these predictions will be used instead of computing them from the model.

    Keywords:
    - bins: The number of bins or the bin edges for the histograms. Default is 1000.
    - n_total: The number of thresholds to evaluate. Default is 200.
    - sample_size: The number of samples to use for computing the mean and standard deviation. Default is 100,000.

    Returns:
    - N_tp: An array containing the number of true positives for each threshold.
    - N_fp: An array containing the number of false positives for each threshold.
    """
    N_tp = np.zeros(n_total)
    N_fp = np.zeros(n_total)
    keep_indices = np.zeros(n_total, dtype=bool)
    prediction_values = np.linspace(-10, 10, n_total)

    if mc_signal_hist is None or mc_background_hist is None:
        use_fit_template = False

    if not use_fit_template:
        def fit_pdf(x, amplitude_sig, sigma_cb, alpha_cb, n_cb, decay_constant):
            return true_data_fit(x, amplitude_sig, 91.188, 2.4955, sigma_cb, alpha_cb, n_cb, decay_constant, x_range=(bins[0], bins[-1]))

    initial_guess = [0.2, 0.1, 1.5, 0.5, 0.05]
    saved_fit_vals = []
    saved_fit_errors = []
    saved_chi2 = []

    if data_truth is not None and mc_signal_limit is not None:
        data_truth = np.array(data_truth, dtype=bool)

        z_predictions = predictions[data_truth][:mc_signal_limit]
        bkg_predictions = predictions[~data_truth]
        comb_predictions = (z_predictions, bkg_predictions)
        predictions = np.concatenate((z_predictions, bkg_predictions))

        z_data_masses = data_masses[data_truth][:mc_signal_limit]
        bkg_data_masses = data_masses[~data_truth]
        data_masses = np.concatenate((z_data_masses, bkg_data_masses))

        z_truth = data_truth[data_truth][:mc_signal_limit]
        bkg_truth = data_truth[~data_truth]
        data_truth = np.concatenate((z_truth, bkg_truth))

    elif data_truth is not None:
        data_truth = np.array(data_truth, dtype=bool)
        z_predictions = predictions[data_truth]
        bkg_predictions = predictions[~data_truth]
        comb_predictions = (z_predictions, bkg_predictions)
    else:
        comb_predictions = None

    t = 0
    for i in range(n_total):
        predicted_labels = (predictions > prediction_values[i]).astype(bool)
        z_masses = data_masses[predicted_labels]
        z_masses = z_masses / 1000
        data_size = len(z_masses)
        if data_size < 1000:
            t += 1
            continue

        fit_exclusion_mask = (z_masses > bins[0]) & (z_masses < bins[-1])
        z_masses = z_masses[fit_exclusion_mask]
        data_fit_size = len(z_masses)
        
        if save_namespace is not None:
            plot_name = f"fits/{save_namespace}_threshold_{i}.png"
            plot_path = f"{plotting_path}/{plot_name}"
        else:
            plot_path = None

        if use_fit_template:
            fit_vals = fit_data_to_mc(z_masses, mc_signal_hist, mc_background_hist, bins, premade_hists=True, save_path=plot_path)
            signal_strength = fit_vals[0]
            saved_fit_vals.append(fit_vals[0])
            saved_fit_errors.append(fit_vals[1])
            saved_chi2.append(fit_vals[2])

        else:
            data_truth_i = (*comb_predictions, prediction_values[i]) if data_truth is not None else None
            fit_vals = fit_function(z_masses, bins, fit_pdf, initial_guess=initial_guess, limits=[(0.0, 1.0), (0.0, 5.0), (0.0, 5.0), (0.0, 10.0), (0.0, 5.0)], save_path=plot_path, data_truth=data_truth_i)
            initial_guess = fit_vals[0]  # Use the fitted values as the initial guess for the next iteration
            signal_strength = initial_guess[0]

            saved_fit_vals.append(fit_vals[0])
            saved_fit_errors.append(fit_vals[1])
            saved_chi2.append(fit_vals[2])

        
        keep_indices[i] = fit_vals[-1] == 1

        N_tp[i] = signal_strength * data_fit_size
        N_fp[i] = (1 - signal_strength) * data_fit_size


    saved_fit_vals = np.array(saved_fit_vals).T
    saved_fit_errors = np.array(saved_fit_errors).T
    saved_chi2 = np.array(saved_chi2)

    parameter_names = ["Amplitude (Signal)", "Mean (BW)", "Width (BW)", "Sigma (CB)", "Alpha (CB)", "n (CB)", "Decay Constant"]
    fig, axes = plt.subplots(saved_fit_vals.shape[0] + 1, 1, figsize=(10, 40))
    axes[0].set_title("Fitted Parameters")
    for j in range(0, saved_fit_vals.shape[0]):
        axes[j].errorbar(range(len(saved_fit_vals[j])), saved_fit_vals[j], yerr=saved_fit_errors[j], label=parameter_names[j], fmt='o', capsize=5)
        axes[j].set_ylabel(f"Parameter {j}")
        axes[j].grid(linestyle='--', alpha=0.5)
    axes[-1].plot(saved_chi2, label="Chi-squared")
    axes[-1].set_ylabel("Chi-squared")
    axes[-1].grid(linestyle='--', alpha=0.5)
    axes[-1].set_xlabel("Threshold Index")
    plt.tight_layout()
    plot_name = f"{save_namespace}_fit_parameters.png"
    plot_path = f"{plotting_path}/{plot_name}"
    save_plot(plot_path, fig)
    plt.close()

    N_tp = N_tp[keep_indices]
    N_fp = N_fp[keep_indices]

    print(f"Fit failed for {n_total - keep_indices.sum() - t} out of {n_total} thresholds.")
    print(f"Fit not initialized for {t} thresholds due to insufficient data points (<1000).")

    return N_tp, N_fp

def _get_cut_totals_from_distribution(
    mc_truth_masses : ArrayLike,
    mc_origin : ArrayLike,
    data_origin : ArrayLike,
    data_masses : ArrayLike,
    mc_signal_hist : ArrayLike | None = None, 
    mc_background_hist : ArrayLike | None = None,
    bins : int | list[float] = np.arange(54.95, 135.05, 0.1),
    plotting_path : str | Path | None = None,
    mc_origin_truth : ArrayLike | None = None,
    mc_signal_limit : int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Finds the total number of true positives and true negatives for different prediction thresholds by fitting the data to the Monte Carlo signal and background histograms.

    Arguments:
    - mc_files_path: A list of paths to the H5 files containing the Monte Carlo data.
    - data_files_path: A list of paths to the H5 files containing the data.
    - mc_masses: The Z boson masses from the Monte Carlo data.
    - data_masses: The Z boson masses from the data.
    - dataloader_kwargs: A dictionary of keyword arguments to pass to the DataLoader constructor
    - valid_fields: A dictionary of needed non_nan fields for the dataset.

    Keywords:
    - bins: The number of bins or the bin edges for the histograms. Default is 1000.
    - mc_signal_hist: The Monte Carlo signal histogram.
    - mc_background_hist: The Monte Carlo background histogram.
    - use_fit_template: A boolean indicating whether to use a fit template. Default is False.
    - predictions_path: A path to a file containing pre-computed predictions. If provided, these predictions will be used instead of computing them from the model.

    Returns:
    - N_tp: An array containing the number of true positives for each threshold.
    - N_fp: An array containing the number of false positives for each threshold.
    """
    if mc_signal_hist is None or mc_background_hist is None:
        use_fit_template = False

    if not use_fit_template:
        def fit_pdf(x, amplitude_sig, sigma_cb, alpha_cb, n_cb, decay_constant):
            return true_data_fit(x, amplitude_sig, 91.188, 2.4955, sigma_cb, alpha_cb, n_cb, decay_constant, x_range=(bins[0], bins[-1]))

    if mc_origin_truth is not None and mc_signal_limit is not None:
        sig_mc_origin = mc_origin[mc_origin_truth][:mc_signal_limit]
        bkg_mc_origin = mc_origin[~mc_origin_truth]
        mc_origin = np.concatenate((sig_mc_origin, bkg_mc_origin))

        sig_mc_truth_masses = mc_truth_masses[mc_origin_truth][:mc_signal_limit]
        bkg_mc_truth_masses = mc_truth_masses[~mc_origin_truth]
        mc_truth_masses = np.concatenate((sig_mc_truth_masses, bkg_mc_truth_masses))


    z_masses_mc = mc_truth_masses[mc_origin] #mc_masses[mc_origin]
    z_masses_mc = z_masses_mc / 1000

    fit_exclusion_mask = (z_masses_mc > bins[0]) & (z_masses_mc < bins[-1])
    z_masses_mc = z_masses_mc[fit_exclusion_mask]
    mc_fit_size = len(z_masses_mc)

    if plotting_path is not None:
        plot_name = f"fits/mc_cut_threshold.png"
        plot_path = f"{plotting_path}/{plot_name}"
    else:
        plot_path = None

    if use_fit_template:
        fit_vals = fit_data_to_mc(z_masses_mc, mc_signal_hist, mc_background_hist, bins, premade_hists=True)
        signal_strength = fit_vals[0]
    else:
        fit_vals = fit_function(z_masses_mc, bins, fit_pdf, initial_guess=[0.9, 1.0, 1.0, 1.0, 0.5], limits=[(0.0, 1.0), (0.0, 5.0), (0.0, 5.0), (0.0, 10.0), (0.0, 5.0)], save_path=plot_path)
        signal_strength = fit_vals[0][0]

    N_tp_mc = signal_strength * mc_fit_size
    N_fp_mc = (1 - signal_strength) * mc_fit_size

    z_masses_data = data_masses[data_origin] #Data does not have known masses. Perhaps predicted masses better than atlas masses?
    z_masses_data = z_masses_data / 1000
    data_size = len(z_masses_data)

    fit_exclusion_mask = (z_masses_data > bins[0]) & (z_masses_data < bins[-1])
    z_masses_data = z_masses_data[fit_exclusion_mask]
    data_fit_size = len(z_masses_data)
    
    if plotting_path is not None:
        plot_name = f"fits/data_cut_threshold.png"
        plot_path = f"{plotting_path}/{plot_name}"
    else:
        plot_path = None

    if use_fit_template:
        fit_vals = fit_data_to_mc(z_masses_data, mc_signal_hist, mc_background_hist, bins, premade_hists=True)
        signal_strength = fit_vals[0]
    else:
        fit_vals = fit_function(z_masses_data, bins, fit_pdf, initial_guess=[0.9, 1.0, 1.0, 1.0, 0.5], limits=[(0.0, 1.0), (0.0, 5.0), (0.0, 5.0), (0.0, 10.0), (0.0, 5.0)], save_path=plot_path)
        signal_strength = fit_vals[0][0]

    N_tp_data = signal_strength * data_fit_size
    N_fp_data = (1 - signal_strength) * data_fit_size

    return N_tp_mc, N_fp_mc, N_tp_data, N_fp_data


def save_plot(path : str | Path, fig : plt.Figure, dpi : int = 300) -> None:
    """
    Save a matplotlib figure to a specified path with a given DPI.
    The function ensures that the parent directory exists before saving the figure.

    Arguments:
        path (str | Path): The path to save the figure to.
        fig (plt.Figure): The matplotlib figure to save.

    Keywords:        
        dpi (int, optional): The DPI of the saved figure. Defaults to 300.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')


def create_pseudo_roc_curve(
    origin_predictions_path : tuple[str] | tuple[Path],
    mass_atlas_path : tuple[str] | tuple[Path],
    origin_atlas_path : tuple[str] | tuple[Path],
    mc_origin_truth_path : str | Path,
    mc_mass_truth_path : str | Path,
    plotting_path : str | Path,
    bins : int | list[float] = 100,
    n_total : int = 200,
    save_namespace : list[str] | list[None] = [None, None],
    template_z_masses_path : tuple[str] | tuple[Path] | None = None,
    mc_signal_limit : int | None = None,
) -> tuple[np.ndarray[int], np.ndarray[int], np.ndarray[int], np.ndarray[int], np.ndarray[int], np.ndarray[int], int, int]:
    """
    Creates a pseudo ROC curve by computing the number of true positives and true negatives for different prediction thresholds, as well as the cut true positives and true negatives. This function is useful for evaluating the performance of a binary classification model.

    Arguments:
    - mc_z_masses_path: A tuple containing the paths to the CSV files with the predicted Z masses for the Monte Carlo (MC) data.
    - mc_files: A list of paths to the Monte Carlo (MC) H5 files
    - data_files: A list of paths to the real data H5 files
    - models: A dictionary containing the PyTorch models used for making predictions. The key "origin" should correspond to the classification model.
    - device: The device (CPU or GPU) on which the models and data are located

    Keywords:
    - batch_size: The number of samples to process in each batch. Default is 4096.
    - num_workers: Number of DataLoader worker processes used for reads. Default is 4
    - prefetch_factor: Number of batches to prefetch for each DataLoader worker. Default is 2.
    - persistent_workers: Whether to keep DataLoader workers alive after the dataset has been consumed.
    - pin_memory: Whether to pin memory for DataLoader workers. Default is True.
    - bins: The number of bins or the bin edges to use for histogramming the predicted Z masses. Default is 1000.
    - n_total: The number of thresholds to evaluate. Default is 200.
    - sample_size: The number of samples to use for computing the mean and standard deviation. Default is 100,000.
    - use_mass_predictions: Whether to use the predicted Z masses for the analysis instead of "egammas" Z masses. Default is True.

    Returns:
    - N_tp_mc: An array containing the true number of true positives for each threshold.
    - N_fp_mc: An array containing the true number of false positives for each threshold.
    - N_tp_fit_mc: An array containing the fitted number of true positives for each threshold.
    - N_fp_fit_mc: An array containing the fitted number of false positives for each threshold.
    - N_tp_fit_data: An array containing the fitted number of true positives for each threshold.
    - N_fp_fit_data: An array containing the fitted number of false positives for each threshold.
    - N_tp_cut: The number of true positives for the cut-based approach.
    - N_fp_cut: The number of false positives for the cut-based approach.
    """
    print(f"Creating pseudo ROC curve with {n_total} thresholds.")
    print(f"Loading the relevant data...")
    mc_origin_predictions = np.loadtxt(origin_predictions_path[0], delimiter=',', dtype=float)
    data_origin_predictions = np.loadtxt(origin_predictions_path[1], delimiter=',', dtype=float)

    #mc_z_masses = np.loadtxt(mass_atlas_path[0], delimiter=',') #Depreciated, I will still have it here in case of later use.
    data_z_masses = np.loadtxt(mass_atlas_path[1], delimiter=',', dtype=float)

    mc_origin_atlas = np.loadtxt(origin_atlas_path[0], delimiter=',', dtype=float) > 0.5
    data_origin_atlas = np.loadtxt(origin_atlas_path[1], delimiter=',', dtype=float) > 0.5

    mc_origin_truth = np.loadtxt(mc_origin_truth_path, delimiter=',', dtype=float) > 0.5
    mc_mass_truth = np.loadtxt(mc_mass_truth_path, delimiter=',', dtype=float)


    print(f"Getting base truth totals for MC files...")
    N_tp_mc, N_fp_mc, N_tp_cut, N_fp_cut = _get_truth_totals(
        mc_origin_predictions,
        mc_origin_truth,
        mc_origin_atlas,
        z_masses_truth = mc_mass_truth,
        n_total=n_total,
        bins=bins,
    )

    if template_z_masses_path:
        print(f"Histogramming the predicted Z masses for precomputed MC files...")
        temp_z_masses = np.loadtxt(template_z_masses_path[0], max_rows=1_000_000)
        sig_z_mass_hist, _ = np.histogram(temp_z_masses / 1000, bins=bins)
        temp_z_masses = np.loadtxt(template_z_masses_path[1], max_rows=1_000_000)
        bkg_z_mass_hist, _ = np.histogram(temp_z_masses / 1000, bins=bins)
    else:
        sig_z_mass_hist, bkg_z_mass_hist = None, None

    print(f"Fitting the MC Z mass distributions...")
    N_tp_fit_mc, N_fp_fit_mc = _get_totals_from_distribution(
        mc_origin_predictions,
        mc_mass_truth,
        data_truth = mc_origin_truth,
        mc_signal_hist = sig_z_mass_hist,
        mc_background_hist = bkg_z_mass_hist,
        bins=bins,
        n_total=n_total,
        plotting_path=plotting_path,
        save_namespace=save_namespace[0],
        mc_signal_limit=mc_signal_limit,
    )

    print(f"Fitting the data Z mass distributions...")
    N_tp_fit_data, N_fp_fit_data = _get_totals_from_distribution(
        data_origin_predictions,
        data_z_masses,
        mc_signal_hist = sig_z_mass_hist,
        mc_background_hist = bkg_z_mass_hist,
        bins=bins,
        n_total=n_total,
        plotting_path=plotting_path,
        save_namespace=save_namespace[1],
    )

    N_tp_cut_mc, N_fp_cut_mc, N_tp_cut_data, N_fp_cut_data = _get_cut_totals_from_distribution(
        mc_mass_truth,
        mc_origin_atlas,
        data_origin_atlas,
        data_z_masses,
        mc_signal_hist = sig_z_mass_hist,
        mc_background_hist = bkg_z_mass_hist,
        bins=bins,
        plotting_path=plotting_path,
        mc_signal_limit=mc_signal_limit,
        mc_origin_truth = mc_origin_truth
    )

    print(f"Pseudo ROC curve creation complete. Plotting results.")
    pseudo_roc_plot_path = Path(plotting_path) / "pseudo_roc_curve.png"
    fig = plt.figure(figsize=(8, 6))
    plt.plot(N_fp_mc/N_fp_cut, N_tp_mc/N_tp_cut, '.', color='blue', label='True pseudo ROC')
    plt.plot(N_fp_fit_mc/N_fp_cut_mc, N_tp_fit_mc/N_tp_cut_mc, '.', color='orange', label='MC Fitted pseudo ROC')
    plt.plot(N_fp_fit_data/N_fp_cut_data, N_tp_fit_data/N_tp_cut_data, '.', color='green', label='Data Fitted pseudo ROC')
    plt.axhline(1, color='black', linestyle='--', label='Cut True Positive Rate')
    plt.axvline(1, color='black', linestyle='--', label='Cut False Positive Rate')
    plt.xlabel('False Positive Rate (Normalized to Cut)')
    plt.ylabel('True Positive Rate (Normalized to Cut)')
    plt.xlim(0.0, 2.0)
    plt.ylim(0.0, 2.0)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    save_plot(pseudo_roc_plot_path, fig)
    plt.close()

    return (N_tp_mc, N_fp_mc), (N_tp_cut, N_fp_cut), (N_tp_fit_mc, N_fp_fit_mc), (N_tp_cut_mc, N_fp_cut_mc), (N_tp_fit_data, N_fp_fit_data), (N_tp_cut_data, N_fp_cut_data)


def get_std_plot(
    predictions : ArrayLike,
    std_predictions : ArrayLike,
    truth_origin : ArrayLike,
    bins : int | list[float] = np.arange(-54.95, 135.05, 0.1),
    plotting_path : str | Path | None = None,
    std_bins : int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the standard deviation of the predictions for different prediction thresholds and plots the results.

    Arguments:
    - predictions: An array of model predictions.
    - prediction_stds: An array of prediction standard deviations.
    - truth_origin: An array of ground truth labels indicating the origin of each sample.

    Keywords:
    - bins: The number of bins or the bin edges for histogramming the predictions. Default is 1000.
    - plotting_path: A path to save the plot. If None, the plot will not be saved.

    Returns:
    - stds: An array containing the standard deviation of the predictions for each threshold.
    - thresholds: An array containing the prediction thresholds.
    """
    quantiles = np.linspace(0.05, 0.95, std_bins + 1)
    quantile_values = np.quantile(std_predictions, quantiles)

    predictions = predictions[truth_origin]
    std_predictions = std_predictions[truth_origin]

    def fit_pdf(x, bw_width, sigma_cb, alpha_cb, n_cb):
        return true_data_fit(x, 1, 91.188, bw_width, sigma_cb, alpha_cb, n_cb, 0, x_range=(bins[0], bins[-1]))

    observed_stds = []
    binned_stds = []
    last_quantile = 0.0
    for quantile in quantile_values:
        if quantile == quantile_values[0]:
            continue 
        if quantile == quantile_values[-1]:
            mask = std_predictions > last_quantile
        else:
            mask = (std_predictions > last_quantile) & (std_predictions <= quantile)
        binned_predictions = predictions[mask]
        quant_mid = (quantile-last_quantile)/2
        guess = [quant_mid, 0.1, 1.5, 0.8]
        save_path = f"{plotting_path}/std_fit/quantile_{quant_mid:.2f}.png" if plotting_path is not None else None

        fitted_params = fit_function(binned_predictions, bins, fit_pdf, initial_guess=guess, limits=[(0.0, 20.0), (0.0, 5.0), (0.0, 5.0), (0.0, 10.0)], save_path=save_path)
        fitted_bw_width_n_error = [fitted_params[0][0], fitted_params[1][0]]
        observed_stds.append(fitted_bw_width_n_error)
        binned_stds.append([quant_mid, quantile - quant_mid])
    observed_stds = np.array(observed_stds)
    binned_stds = np.array(binned_stds)

    if plotting_path is not None:
        std_plot_path = Path(plotting_path) / "std_fit_results.png"
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(binned_stds[:, 0], observed_stds[:, 0], yerr=observed_stds[:, 1], xerr=binned_stds[:, 1], fmt='o', capsize=5)
        ax.set_xlabel('Prediction Standard Deviation Quantile Midpoint')
        ax.set_ylabel('Fitted BW Width (GeV)')
        ax.set_title('Fitted BW Width vs. Prediction Standard Deviation Quantiles')
        ax.grid(True, linestyle='--', alpha=0.7)
        save_plot(std_plot_path, fig)
        plt.close()

    return observed_stds, binned_stds



