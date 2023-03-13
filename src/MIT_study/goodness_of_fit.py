import numpy as np
from scipy.stats import gumbel_r, expon, gumbel_r, gumbel_l, lognorm, logistic, norm, weibull_min
from scipy.stats import kstest, anderson, chi2, probplot

def goodness_of_fits(g):
    """find goodness of fits of different specified distributions"""
    print(f'Calculating stats on station {g.name}')
    data = g['wspd_merge']

    # Define the candidate distributions
    dist_labels = ['exponential', 'gumbel_l', 'gumbel_r', 'log_norm', 'logistic', 'norm']
    distributions = [expon, gumbel_l, gumbel_r, lognorm, logistic, norm]

    # Perform the goodness-of-fit test for each distribution
    for label, dist in zip(dist_labels, distributions):
        #print(f'Fitting {label} distribution')

        # Fit the distribution to the data
        params = dist.fit(data)
        
        # Calculate the KS test statistic and p-value
        # Kolmogorov-Smirnov
        ks_stat, ks_pvalue = kstest(data, dist.cdf, args=params)
        
        # Calculate the Anderson-Darling test statistic and critical values
        ad_stat, _, ad_critvalues = anderson(data, dist.name)
        
        # Calculate the chi-squared test statistic and p-value
        hist, bin_edges = np.histogram(data, bins='auto', density=True)
        cdf_observed = np.cumsum(hist * np.diff(bin_edges))
        cdf_expected_bins = dist.cdf(bin_edges, *params)
        chi2_stat = np.sum((cdf_observed - cdf_expected_bins[:-1]) ** 2 / cdf_expected_bins[:-1])
        chi2_pvalue = chi2.sf(chi2_stat, len(data) - len(params))
        
        # Print the test results
        print(f"Test results for {dist.name}")
        print(f"KS test statistic: {ks_stat:.4f}, p-value: {ks_pvalue:.4f}")
        print(f"Anderson-Darling test statistic: {ad_stat:.4f}, critical values: {ad_critvalues}")
        print(f"Chi-squared test statistic: {chi2_stat:.4f}, p-value: {chi2_pvalue:.4f}\n")

def prob_plot_correlation_coeff(g):
    """
    Probability plot correlation coefficient for a station
    
    Return:
        r_squared (float): correlation coefficient
    """

    print(f'Calculating stats on station {g.name}')
    data = g['wspd_merge']

    # Fit the data to the exponential distribution and get the parameter
    lam = expon.fit(data)[0]

    # Generate a probability plot and calculate the correlation coefficient
    probplot_data = probplot(data, dist=expon, sparams=(lam,))
    exp_r_squared = probplot_data[1][0]

    # Fit the data to the Gumbel distribution and get the parameters
    (loc, scale) = gumbel_r.fit(data)

    # Generate a probability plot and calculate the correlation coefficient
    probplot_data = probplot(data, dist=gumbel_r, sparams=(loc, scale))
    gum_r_squared = probplot_data[1][0]

    # Fit the data to the log-normal distribution and get the parameters
    (shape, loc, scale) = lognorm.fit(data)

    # Generate a probability plot and calculate the correlation coefficient
    probplot_data = probplot(data, dist=lognorm, sparams=(shape, loc, scale))
    ln_r_squared = probplot_data[1][0]

    # Fit the data to the Weibull distribution and get the parameters
    (c, loc, scale) = weibull_min.fit(data)

    # Generate a probability plot and calculate the correlation coefficient
    probplot_data = probplot(data, dist=weibull_min, sparams=(c, loc, scale))
    wb_r_squared = probplot_data[1][0]

    return {'exp_r': exp_r_squared,
                'gumbel_r': gum_r_squared, 
                'lognorm':ln_r_squared,
                'weibull_min': wb_r_squared
     }
