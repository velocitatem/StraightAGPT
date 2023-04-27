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
    You must have a dataset. For a lot of problems you will need to understand some dataset. This tool will give you a summary of the dataset. Params: path/link,format. Example input is: https://raw.githubusercontent.com/llimllib/bostonmarathon/master/results/2014/results.csv,csv
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
def chi_squared_metric(comma_separated_list):
    '''
    Calculates the chi-squared metric. Params: n,standard_deviation_sample,standard_deviation_null. Example input is: 100,1,0.7
    '''
    comma_separated_list = [x.strip() for x in comma_separated_list.split(',')]
    n, standard_deviation, standard_deviation_0 = map(float, comma_separated_list)
    # ¦((Q-1)*Vd**2)/(Vd0**2)
    chi_squared = ((n-1)*(standard_deviation**2))/(standard_deviation_0**2)


    return chi_squared

@tool
def tail_specific_chi_squared_value(comma_separated_list):
    '''
    Returns the p-value for the specific tail of the chi-squared-value. Params: chi-squared-value, tail, degrees_of_freedom. Example input is: 2.4,left/right/two,99(n-1)
    '''
    comma_separated_list = [x.strip() for x in comma_separated_list.split(',')]
    chi_squared, tail, degrees_of_freedom = map(str, comma_separated_list)
    chi_squared = float(chi_squared)
    degrees_of_freedom = float(degrees_of_freedom)
    if (tail == "left"):
        return stats.chi2.cdf(chi_squared, degrees_of_freedom)
    elif (tail == "right"):
        return 1 - stats.chi2.cdf(chi_squared, degrees_of_freedom)
    elif (tail == "two"):
        return 2 * (1 - stats.chi2.cdf(abs(chi_squared), degrees_of_freedom))
    else:
        return "Error: tail must be left, right or two"

@tool
def test_statistic_selection(problem):
    '''
    Returns a test statistic for a specific problem. You should always provide the entire context and the problem
    '''
    llm = OpenAI(model_name="text-davinci-003")
    prompt = f'''
    Rules:
    if proportions:
        return proportions_z_score
    if variation or standard deviation:
        return chi_squared
    if mean:
        if n>30:
            return z_score
        else:
            return t_score
    ---
    Problem: {problem}
    Test statistic:'''
    response = llm(prompt.format(problem=problem))
    return response


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

@tool
def t_confidence_interval(comma_separated_list):
    '''
    Calculates the t confidence interval. Params: sample_mean,standard_deviation,n,alpha. Example input is: 158,1,100,0.05
    '''
    comma_separated_list = [x.strip() for x in comma_separated_list.split(',')]
    sample_mean, standard_deviation, n, alpha = map(float, comma_separated_list)
    t = stats.t.ppf(1 - alpha / 2, n - 1)
    lower_bound = sample_mean - t * standard_deviation / np.sqrt(n)
    upper_bound = sample_mean + t * standard_deviation / np.sqrt(n)
    return lower_bound, upper_bound

@tool
def z_confidence_interval(comma_separated_list):
    '''
    Calculates the z confidence interval. Params: sample_mean,standard_deviation,n,alpha. Example input is: 57,1,100,0.05
    '''
    comma_separated_list = [x.strip() for x in comma_separated_list.split(',')]
    sample_mean, standard_deviation, n, alpha = map(float, comma_separated_list)
    z = stats.norm.ppf(1 - alpha / 2)
    lower_bound = sample_mean - z * standard_deviation / np.sqrt(n)
    upper_bound = sample_mean + z * standard_deviation / np.sqrt(n)
    return lower_bound, upper_bound

@tool
def proportions_z_confidence_interval(comma_separated_list):
    '''
    Calculates the proportions z confidence interval. Params: sample_proportion,n,alpha. Example input is: 0.5,100,0.05
    '''
    comma_separated_list = [x.strip() for x in comma_separated_list.split(',')]
    sample_proportion, n, alpha = map(float, comma_separated_list)
    z = stats.norm.ppf(alpha / 2)
    lower_bound = sample_proportion - z * np.sqrt(sample_proportion * (1 - sample_proportion) / n)
    upper_bound = sample_proportion + z * np.sqrt(sample_proportion * (1 - sample_proportion) / n)
    return lower_bound, upper_bound



@tool
def select_confidence_interval(comma_separated_list):
    '''
    Selects the confidence interval based on the sample size. Params: n. Example input is: 100
    '''
    comma_separated_list = [x.strip() for x in comma_separated_list.split(',')]
    n = float(comma_separated_list[0])
    if n > 30:
        return 'z_confidence_interval'
    else:
        return 't_confidence_interval'

def list_tools():
    return [
        calculate_beta,
        z_score, # z testing
        tail_specific_z_value,
        t_score, # t testing
        tail_specific_t_value,
        proportions_z_score, # proportions testing
        tail_specific_proportions_z_value,
        chi_squared_metric, # chi-squared testing
        tail_specific_chi_squared_value,
        test_statistic_selection,
        required_sample_size,
        t_confidence_interval, # confidence intervals
        z_confidence_interval,
        proportions_z_confidence_interval,
        select_confidence_interval,
        load_dataset
            ]
