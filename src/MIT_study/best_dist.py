import numpy as np
from typing import Union
from scipy.stats import anderson

def does_data_follow_dist(data: np.ndarray,
                          candidate_dist: str,
                          desired_significance_level: float,
                          statistical_test: str = 'anderson_darling') -> Union[bool, None]:
    """
    Performs a statistical test to ascertain if the data follows a particular distribution.

    All distributional tests calculate critical values assuming that the data follows the candidate distribution. Thus,
    if the test statistic exceeds the critical value associated with the chosen significance level, we can reject the
    null hypothesis that the data follows the candidate distribution. However, if the test statistic does NOT exceed the
    critical value associated with the chosen significance level, we cannot necessarily conclude that the data
    follows the candidate distribution.
    Therefore, we return 'False' when we reject the null but return 'None' when the test is not able to reject the null hypothesis.
    
    Args:
        data: the data whose distribution we wish to ascertain.
        candidate_dist: the candidate distribution which we want to test for.
        desired_significance_level: the desired significance level between 0 and 1.
        statistical_test: the type of statistical test we wish to perform. Default: anderson_darling
    
    Returns:
        bool or None: False if the data does NOT follow the candidate distribution, None otherwise.
    """
    # ****** SANITY CHECKS ON INPUT ************************************************************************************
    supported_statistical_tests = ['anderson_darling']
    if statistical_test not in supported_statistical_tests:
        tests_str = ', '.join(supported_statistical_tests)
        error_msg = '{statistical_test} is NOT supported.\n'.format(statistical_test=statistical_test) + 'Use one of: {tests_str}'.format(tests_str=tests_str)

        raise ValueError(error_msg)

    # Note: Maintain list in alphabetical order.
    supported_candidate_dist = ['exponential', 'gumbel_l', 'gumbel_r', 'log_norm', 'logistic', 'norm']
    if candidate_dist not in supported_candidate_dist:
        dist_str = ', '.join(supported_candidate_dist)
        error_msg = '{candidate_dist} is NOT supported.\n'.format(candidate_dist=candidate_dist) + 'Use one of: {dist_str}'.format(dist_str=dist_str)

        raise ValueError(error_msg)

    if desired_significance_level < 0 or desired_significance_level > 1:
        error_msg = 'The significance level passed via "desired_significance_level" must be between 0 and 1.'
        raise ValueError(error_msg)

    # The anderson function in scipy.stats computes different significance levels depending on distribution.
    supported_significance_levels = {
        'log_norm': [0.15, 0.10, 0.05, 0.025, 0.01],
        'norm': [0.15, 0.10, 0.05, 0.025, 0.01],
        'exponential': [0.15, 0.10, 0.05, 0.025, 0.01],
        'logistic': [0.25, 0.10, 0.05, 0.025, 0.01, 0.05],
        'gumbel_l': [0.25, 0.10, 0.05, 0.025, 0.01],
        'gumbel_r': [0.25, 0.10, 0.05, 0.025]
    }

