import scipy.stats as stats
import statsmodels.stats.power as ssp
from statsmodels.stats.power import TTestPower
import numpy as np
from langchain.agents import tool
from langchain.llms import OpenAI


@tool
def calculate_beta(comma_separated_list):
    '''Use only if explicitly asked to. Calculates Beta, the probability of the type II error. Uses sample_mean,population_mean,n,s and returns beta (1-power). Example input is: 0.5,0,100,1'''
    sample_mean, population_mean, n, s = map(float, comma_separated_list.split(','))
    print(sample_mean, population_mean, n, s)
    if n > 30:
        power=ssp.normal_power_het(population_mean - sample_mean, n, 0.05, std_null=s, std_alternative=None, alternative='larger')
    else:
        power = TTestPower().power(effect_size=population_mean - sample_mean, nobs=n, alpha=0.05, alternative='large')
    return 1-power


@tool
def z_score(comma_separated_list):
    '''Calculates z-test. Returns z-value. Params: sample_mean,population_mean,n,s. Example input is: 0.5,0,100,1'''
    sample_mean, population_mean, n, s = map(float, comma_separated_list.split(','))
    z = (sample_mean - population_mean) / (s / np.sqrt(n)) # TODO should this be just s?
    return z

@tool
def tail_specific_z_value(comma_separated_list):
    """Returns the p-value for the specific tail of the z-value. Params: z-value, tail. Example input is: 0.5,left/right/two"""
    z, tail = map(str, comma_separated_list.split(','))
    z = float(z)
    if (tail == "left"):
        return stats.norm.cdf(z)
    elif (tail == "right"):
        return 1 - stats.norm.cdf(z)
    elif (tail == "two"):
        return 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        return "Error: tail must be left, right or two"

@tool
def t_score(comma_separated_list):
    '''
    Calculates the t-score. Params: sample_mean,population_mean,n,s. Example input is: 0.5,0,100,1
    '''
    sample_mean, population_mean, n, s = map(float, comma_separated_list.split(','))
    t = (sample_mean - population_mean) / (s / np.sqrt(n))
    return t


@tool
def tail_specific_t_value(comma_separated_list):
    '''
    Returns the p-value for the specific tail of the t-value. Params: t-value, tail, n. Example input is: 0.5,left/right/two,100
    '''
    t, tail, n = map(str, comma_separated_list.split(','))
    t = float(t)
    if (tail == "left"):
        return stats.t.cdf(t, n - 1)
    elif (tail == "right"):
        return 1 - stats.t.cdf(t, n - 1)
    elif (tail == "two"):
        return 2 * (1 - stats.t.cdf(abs(t), n - 1))
    else:
        return "Error: tail must be left, right or two"



@tool
def test_statistic_selection(params):
    '''Selects the appropriate test statistic based on the number of samples. Params: size,sigma. Example input is: 100,1'''
    n, s = map(float, params.split(','))
    options = ['z-test', 't-test']
    if s is None:
        return options[1]
    else:
        if n > 30:
            return options[0]
        else:
            return options[1]



def list_tools():
    return [calculate_beta,
            z_score, # z testing
            tail_specific_z_value,
            t_score, # t testing
            tail_specific_t_value,
            test_statistic_selection]
