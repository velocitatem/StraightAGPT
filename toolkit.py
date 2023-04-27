import scipy.stats as stats
import statsmodels.stats.power as ssp
from statsmodels.stats.power import TTestPower
import numpy as np
from langchain.agents import tool
from langchain.llms import OpenAI
import pandas as pd


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
    comma_separated_list = [x.strip() for x in comma_separated_list.split(',')]
    z, tail = map(str, comma_separated_list)
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
    comma_separated_list = [x.strip() for x in comma_separated_list.split(',')]
    sample_mean, population_mean, n, s = map(float, comma_separated_list)
    t = (sample_mean - population_mean) / (s / np.sqrt(n))
    return t


@tool
def tail_specific_t_value(comma_separated_list):
    '''
    Returns the p-value for the specific tail of the t-value. Params: t-value, tail, n. Example input is: 0.5,left/right/two,100
    '''
    comma_separated_list = [x.strip() for x in comma_separated_list.split(',')]
    t, tail, n = map(str, comma_separated_list)
    t = float(t)
    n = float(n)
    if (tail == "left"):
        return stats.t.cdf(t, n - 1)
    elif (tail == "right"):
        return 1 - stats.t.cdf(t, n - 1)
    elif (tail == "two"):
        return 2 * (1 - stats.t.cdf(abs(t), n - 1))
    else:
        return "Error: tail must be left, right or two"

@tool
def load_dataset(comma_separated_list):
    '''
    For a lot of problems you will need to understand some dataset. This tool will give you a summary of the dataset. Params: path/link,format. Example input is: https://raw.githubusercontent.com/llimllib/bostonmarathon/master/results/2014/results.csv,csv
    '''
    comma_separated_list = [x.strip() for x in comma_separated_list.split(',')]
    path, form = map(str, comma_separated_list)
    df = pd.DataFrame()
    if form == "csv":
        df = pd.read_csv(path)
    elif form == "json":
        df = pd.read_json(path)
    elif form == "excel":
        df = pd.read_excel(path)
    else:
        return "Error: format must be csv, json or excel"

    description = df.describe()
    return description

@tool
def proportions_z_score(comma_separated_list):
    '''
    Calculates the z-score for proportions. Params: null_hypothesis_proportion,sample_proportion,n. Example input is: 0.5,0,100
    '''
    comma_separated_list = [x.strip() for x in comma_separated_list.split(',')]
    null_hypothesis_proportion, sample_proportion, n = map(float, comma_separated_list)
    z = (sample_proportion - null_hypothesis_proportion) / np.sqrt((null_hypothesis_proportion * (1 - null_hypothesis_proportion)) / n)
    return z

@tool
def tail_specific_proportions_z_value(comma_separated_list):
    '''
    Returns the p-value for the specific tail of the z-value for proportions. Params: z-value, tail. Example input is: 0.5,left/right/two
    '''
    comma_separated_list = [x.strip() for x in comma_separated_list.split(',')]
    z, tail = map(str, comma_separated_list)
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
def test_statistic_selection(comma_separated_list):
    '''z-score or t-score. Selects the appropriate test statistic based on the number of samples. Params: size,sigma. Example input is: 100,1'''
    comma_separated_list = [x.strip() for x in comma_separated_list.split(',')]
    n, s = map(float, comma_separated_list)
    options = ['z-test', 't-test', 'proportions-z-test']
    if s is None:
        return options[1]
    else:
        if n > 30:
            return options[0] + " or " + options[2]
        else:
            return options[1]

@tool
def required_sample_size(comma_separated_list):
    '''
    Calculates the required sample size for a specific alpha, beta. Params: alpha,beta,sigma,mean_0,sample_mean. Example input is: 0.05,0.2,1,0,0.5
    '''
    comma_separated_list = [x.strip() for x in comma_separated_list.split(',')]
    alpha, beta, s, mean_0, sample_mean = map(float, comma_separated_list)
    effect_size = (sample_mean - mean_0) / s
    n = (stats.norm.ppf(1 - alpha) + stats.norm.ppf(1 - beta)) ** 2 * s ** 2 / effect_size ** 2
    return n


def list_tools():
    return [
        calculate_beta,
        z_score, # z testing
        tail_specific_z_value,
        t_score, # t testing
        tail_specific_t_value,
        proportions_z_score, # proportions testing
        tail_specific_proportions_z_value,
        test_statistic_selection,
        load_dataset
            ]
