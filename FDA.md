# Problem Selection Tree

Input Variables: `sample_size, sample_sd_dist, sample_variance_dist, sample_mean, population_mean, sample_prop, population_prop, sample_sd, population_sd, alpha, beta`

+ One Sample Confidence Intervals
    + One Sample Proportion
    + One Sample Mean
    + One Sample Variance
+ Hypothesis Testing
  + One Sample
    + One Sample Proportion
    + One Sample Mean
    + One Sample Variance
  + Two Sample
    + Two Sample Proportion
    + Two Sample Mean
    + Two Sample Variance


CONFIDENCE INTERVALS
 - ONE SAMPLE
   - Proportion: Categorical variable, sample size (n) * sample proportion (p) >= 5 and n * (1 - p) >= 5
   - Mean: Quantitative variable, population standard deviation (sigma) known
   - Variance: Chi-squared distribution
 - TWO SAMPLE
   - Proportion: Two categorical variables, each sample size (n1, n2) * sample proportion (p1, p2) >= 5 and n1 * (1 - p1) >= 5 and n2 * (1 - p2) >= 5
   - Mean: Two quantitative variables, population standard deviation (sigma1, sigma2) known or sample sizes (n1, n2) >= 30
   - Variance: F-distribution, two independent samples

HYPOTHESIS TESTING
 - ONE SAMPLE
   - Proportion: Categorical variable, sample size (n) * sample proportion (p) >= 5 and n * (1 - p) >= 5, null hypothesis is p0
   - Mean: Quantitative variable, population standard deviation (sigma) unknown or n >= 30, null hypothesis is mu0
   - Variance: Chi-squared distribution, null hypothesis is sigma^2 = sigma0^2
 - TWO SAMPLE
   - Proportion: Two categorical variables, each sample size (n1, n2) * sample proportion (p1, p2) >= 5 and n1 * (1 - p1) >= 5 and n2 * (1 - p2) >= 5, null hypothesis is p1 = p2
   - Mean: Two quantitative variables, population standard deviation (sigma1, sigma2) unknown or sample sizes (n1, n2) >= 30, null hypothesis is mu1 - mu2 = d0
   - Variance: F-distribution, two independent samples, null hypothesis is sigma1^2 / sigma2^2 = f0
   - Paired: Two quantitative variables, paired differences, null hypothesis is muD = d0

Python:

Decision tree given inputs
```python
if
