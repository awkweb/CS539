# Section B: Univariate Data (175 points + bonus points)
# https://cran.r-project.org/doc/manuals/R-intro.pdf

# I. Data Generation:
set.seed(1)
n = 1000
mean = 60
standard_deviation = 8
X = rnorm(n=n, mean=mean, sd=standard_deviation)


# II. MLE:
# 1
library(stats4)
log_likelihood = function(mu, sigma) {
  R = dnorm(X, mu, sigma, log=TRUE)
  #
  -sum(log(R))
}
mle(log_likelihood, start = list( mu = 1, sigma=1))

# 2


# III. MAP and Bayes' Estimator:


# IV. Classification:
# 1
sample_1 = rnorm(n=500, mean=60, sd=8)
sample_2 = rnorm(n=300, mean=30, sd=12)
sample_3 = rnorm(n=300, mean=80, sd=4)
X = c(sample_1, sample_2, sample_3)

# 2















