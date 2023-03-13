from scipy.stats import gumbel_r, weibull_min, rayleigh, lognorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from multiprocessing import Pool
import pickle
import pyarrow.parquet as pq


def fit(g):
    """
    Fit wind speed using different distributions and plot the distribution fits"""
    g["wspd_merge"] = g["wspd_merge"].astype("float64")
    params_gumbel = gumbel_r.fit(g["wspd_merge"])
    params_weibull = weibull_min.fit(g["wspd_merge"])
    params_lognorm = lognorm.fit(g["wspd_merge"])
    params_rayleigh = rayleigh.fit(g["wspd_merge"])

    return {
        "gumbel": params_gumbel,
        "weibull": params_weibull,
        "lognorm": params_lognorm,
        "rayleigh": params_rayleigh,
    }


def find_magnitudes(param_dict, return_period=100):
    """
    Calculate the magnitude corresponding to a probability exceeding 99%
    (e.g., a probability of 0.01 and a return period of 100 years) and
    a return period using the inverse CDF
    """
    p_exceed = 1 - 1 / return_period

    magnitude_gumbel = gumbel_r.ppf(p_exceed, *param_dict["gumbel"])
    magnitude_weibull = weibull_min.ppf(p_exceed, *param_dict["weibull"])
    magnitude_lognorm = lognorm.ppf(p_exceed, *param_dict["lognorm"])
    magnitude_rayleigh = rayleigh.ppf(p_exceed, *param_dict["rayleigh"])

    return {
        "gumbel": magnitude_gumbel,
        "weibull": magnitude_weibull,
        "lognorm": magnitude_lognorm,
        "rayleigh": magnitude_rayleigh,
    }


def print_magnitudes(magnitude_dict):
    """print magnitude corresponding to a probability exceeding 99%"""
    for key, value in magnitude_dict.items():
        print(
            f"Magnitude of a 100-year event from a {key} distribution:"
            f" {[value]:.3f}"
        )


def find_pdf(param_dict):

    x = np.linspace(
        gumbel_r.ppf(0.001, *param_dict["gumbel"]),
        gumbel_r.ppf(0.999, *param_dict["gumbel"]),
        1000,
    )
    pdf_gumbel = gumbel_r.pdf(x, *param_dict["gumbel"])
    pdf_weibull = weibull_min.pdf(x, *param_dict["weibull"])
    pdf_lognormal = weibull_min.pdf(x, *param_dict["lognorm"])
    pdf_rayleigh = weibull_min.pdf(x, *param_dict["rayleigh"])

    return {
        "gumbel": pdf_gumbel,
        "weibull": pdf_weibull,
        "lognorm": pdf_lognormal,
        "rayleigh": pdf_rayleigh,
    }, x


def plot_fits(x, g, pdf_dict, linewidth=3):
    plt.plot(
        x,
        pdf_dict["gumbel"],
        label="Gumbel fit",
        color="r",
        linewidth=linewidth,
    )
    plt.plot(
        x,
        pdf_dict["weibull"],
        label="Weibull fit",
        color="b",
        linewidth=linewidth,
    )
    plt.plot(
        x,
        pdf_dict["lognorm"],
        label="Weibull fit",
        color="g",
        linewidth=linewidth,
    )
    plt.plot(
        x,
        pdf_dict["rayleigh"],
        label="Weibull fit",
        color="g",
        linewidth=linewidth,
    )

    # plot the raw data as a histogram
    plt.hist(
        g, bins=10, density=True, color="k", alpha=0.5, label="Mean 5 min wspd"
    )
    plt.legend()
    plt.show()


def run_station(g, plot_fits=False):
    # print("fitting station",flush = True)
    param_dict = fit(g)
    # print("finding pdf",flush = True)
    pdf_dict, x = find_pdf(param_dict)
    # print("finding extreme magnitude",flush = True)
    magnitude_dict = find_magnitudes(param_dict)
    # print(magnitude_dict)
    # print_magnitudes(magnitude_dict)
    if plot_fits:
        plot_fits(x, g, pdf_dict)
    return magnitude_dict


def dump_output(magnitude_dict):
    with open(
        "/home/vanessa/hulk/MIT_study/data/output_dict_magnitudes.pkl", "wb"
    ) as fp:
        pickle.dump(magnitude_dict, fp)
        print("dictionary saved successfully to file")


def applyParallel(dfGrouped, func):

    p = Pool(30)
    magnitude_dict = p.map(func, [group for name, group in dfGrouped])
    p.close()

    # magnitude_dict = Parallel(n_jobs=32)(
    #     delayed(func)(group) for name, group in dfGrouped
    # )
    return magnitude_dict       


def main():
    filename = (
        "/home/vanessa/hulk/MIT_study/data/mesonet_wind_2015_2022.parquet.gzip"
    )
    df = pq.read_table(filename).to_pandas().dropna()
    print("done reading in the data")
    df["wspd_merge"] = df["wspd_merge"] * 2.23694  # convert to mph
    print("done converting to mph")
    # df[["station", "wspd_merge"]].groupby(["station"]).apply(run_station)

    magnitude_dict = applyParallel(
        df[["station", "wspd_merge"]].groupby(["station"]), run_station
    )
    dump_output(magnitude_dict)


# if __name__ == "__main__":
#     main()
