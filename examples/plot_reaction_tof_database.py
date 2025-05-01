# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ase_ml_models.utilities import modify_name
from ase_ml_models.workflow import uncertainty_plot, calibrate_uncertainty

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():
    
    # Reaction.
    reaction = "RWGS" # WGS | RWGS

    # Read microkinetics results.
    yaml_results = f"results_database_{reaction}.yaml"
    with open(yaml_results, 'r') as fileobj:
        results_all = yaml.safe_load(fileobj)
    results_DFT = results_all["DFT+DFT"]
    results_BEP = results_all["TSR+BEP"]
    results_WWL = results_all["WWLGPR+WWLGPR"]
    
    # DFT data.
    ltofs_DFT = []
    surfaces_DFT = []
    for jj, surface in enumerate(results_DFT):
        ltofs_DFT.append(np.log10(results_DFT[surface][None]["TOF"]))
        surfaces_DFT.append(modify_name(surface))
    
    # BEP data.
    ltofs_BEP = []
    lstds_BEP = []
    surfaces_BEP = []
    for jj, surface in enumerate(results_BEP):
        tof_list = [results_BEP[surface][ii]["TOF"] for ii in range(5)]
        tof_list = [tof if tof > 1e-20 else 1e-20 for tof in tof_list]
        ltof_list = [np.log10(tof) for tof in tof_list]
        ltofs_BEP.append(np.mean(ltof_list))
        lstds_BEP.append(np.std(ltof_list))
        surfaces_BEP.append(modify_name(surface))

    # WWL-GPR data.
    ltofs_WWL = []
    lstds_WWL = []
    surfaces_WWL = []
    for jj, surface in enumerate(results_WWL):
        tof_list = [results_WWL[surface][ii]["TOF"] for ii in range(5)]
        tof_list = [tof if tof > 1e-20 else 1e-20 for tof in tof_list]
        ltof_list = [np.log10(tof) for tof in tof_list]
        ltofs_WWL.append(np.mean(ltof_list))
        lstds_WWL.append(np.std(ltof_list))
        surfaces_WWL.append(modify_name(surface))

    # Turnover frequencies plot.
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.bar(
        x=surfaces_DFT,
        height=np.array(ltofs_DFT)+14,
        color="orange",
        label="DFT",
        bottom=-14,
    )
    ax.errorbar(
        x=surfaces_BEP,
        y=ltofs_BEP,
        yerr=np.array(lstds_BEP).T,
        fmt="o",
        color="darkcyan",
        label="TSR+BEP",
        capsize=5,
    )
    ax.errorbar(
        x=surfaces_WWL,
        y=ltofs_WWL,
        yerr=np.array(lstds_WWL).T,
        fmt="o",
        color="crimson",
        label="WWL-GPR",
        capsize=5,
    )
    ax.set_xlim(-0.5, len(ltofs_DFT)-0.5)
    ax.set_ylabel("log$_{10}$(TOF [1/s])")
    ax.tick_params(axis="x", rotation=90)
    ax.legend(edgecolor="black")
    plt.subplots_adjust(bottom=0.25, top=0.95, left=0.10, right=0.95)
    dirname = f"images/reaction_{reaction}_database"
    os.makedirs(dirname, exist_ok=True)
    plt.savefig(f"{dirname}/materials_TOF.png", dpi=300)
    
    # MAE and RMSE calculation.
    mae_BEP = mean_absolute_error(ltofs_DFT, ltofs_BEP)
    rmse_BEP = mean_squared_error(ltofs_DFT, ltofs_BEP, squared=False)
    mae_WWL = mean_absolute_error(ltofs_DFT, ltofs_WWL)
    rmse_WWL = mean_squared_error(ltofs_DFT, ltofs_WWL, squared=False)
    print(f"TSR+BEP MAE:   {mae_BEP:7.4f} [log10(1/s)]")
    print(f"TSR+BEP RMSE:  {rmse_BEP:7.4f} [log10(1/s)]")
    print(f"WWL-GPR MAE:   {mae_WWL:7.4f} [log10(1/s)]")
    print(f"WWL-GPR RMSE:  {rmse_WWL:7.4f} [log10(1/s)]")
    
    # Parity plot.
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.errorbar(
        ltofs_DFT,
        ltofs_BEP,
        yerr=lstds_BEP,
        fmt="o",
        color="darkcyan",
        label=f"TSR+BEP\nMAE = {mae_BEP:4.2f}"+" [log$_{10}$(1/s)]",
        capsize=5,
    )
    ax.errorbar(
        ltofs_DFT,
        ltofs_WWL,
        yerr=lstds_WWL,
        fmt="o",
        color="crimson",
        label=f"WWL-GPR\nMAE = {mae_WWL:4.2f}"+" [log$_{10}$(1/s)]",
        capsize=5,
    )
    ax.plot([-14, +1], [-14, +1], color="black", linestyle="--")
    ax.set_xlabel("log$_{10}$(TOF [1/s]) DFT")
    ax.set_ylabel("log$_{10}$(TOF [1/s]) model")
    ax.set_xlim(-14, +1)
    ax.set_ylim(-14, +1)
    plt.subplots_adjust(bottom=0.10, top=0.95, left=0.15, right=0.95)
    plt.legend(edgecolor="black", loc="upper left")
    
    y_err_BEP = np.abs(np.array(ltofs_DFT)-np.array(ltofs_BEP))
    y_err_WWL = np.abs(np.array(ltofs_DFT)-np.array(ltofs_WWL))
    ax = fig.add_axes([0.67, 0.15, 0.25, 0.28])
    violins = ax.violinplot(
        dataset=[y_err_BEP, y_err_WWL],
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.5,
        points=1000,
    )["bodies"]
    violins[0].set_facecolor("darkcyan")
    violins[0].set_edgecolor("black")
    violins[0].set_alpha(1.0)
    violins[1].set_facecolor("crimson")
    violins[1].set_edgecolor("black")
    violins[1].set_alpha(1.0)
    ax.set_ylabel("Errors [log$_{10}$(1/s)]")
    ax.set_ylim([0., 4.5])
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["TSR+BEP", "WWL-GPR"])
    ax.tick_params(width=1.0, length=6, direction="inout")
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    plt.savefig(f"{dirname}/parity_TOF.png", dpi=300)
    
    # Uncertainty plot TSR+BEP.
    results = {"y_err": y_err_BEP, "y_std": lstds_BEP}
    results = calibrate_uncertainty(results=results, n_bins=6, fit_intercept=False)
    ax = uncertainty_plot(
        results=results,
        color="darkcyan",
        lims=[0., 5],
        n_bins=6,
    )
    plt.savefig(f"{dirname}/uncertainty_TSR+BEP.png", dpi=300)
    
    # Uncertainty plot WWL-GPR.
    results = {"y_err": y_err_WWL, "y_std": lstds_WWL}
    results = calibrate_uncertainty(results=results, n_bins=6, fit_intercept=False)
    ax = uncertainty_plot(
        results=results,
        color="crimson",
        lims=[0., 5.],
        n_bins=6,
    )
    plt.savefig(f"{dirname}/uncertainty_WWL-GPR.png", dpi=300)

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------