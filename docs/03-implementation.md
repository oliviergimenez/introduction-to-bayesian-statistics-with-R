# Practical implementation {#software}

## Introduction

In this chapter, we will explore two very practical tools for performing Bayesian statistics with minimal effort: `NIMBLE` and `brms`. `NIMBLE` and `brms` are two `R` packages that implement MCMC algorithms for you. In practice, you only need to specify a likelihood and priors for Bayes’ theorem to be applied automatically. Thanks to a syntax close to that of `R`, both packages make this step relatively straightforward, even for complex models.

## `NIMBLE`

`NIMBLE` stands for **N**umerical **I**nference for statistical **M**odels using **B**ayesian and **L**ikelihood **E**stimation. The originality of `NIMBLE` is that it separates the model-building step from the model-fitting step, which allows great flexibility in modeling. The package is developed by a team of scientists who continuously improve its capabilities based on community feedback. The `NIMBLE` community is active on <https://groups.google.com/g/nimble-users>, a forum where the developers respond to questions quickly and helpfully.

To use `NIMBLE`, you can follow these steps:

1. Build a model (likelihood and priors).
2. Read in the data.
3. Specify the parameters for which you want to make inferences.
4. Provide initial values for these parameters (per chain).
5. Define the MCMC settings: number of chains, burn-in, number of post-burn-in iterations.
6. Assess convergence.
7. Interpret the results.

But first, don’t forget to load the package (to install `NIMBLE`, see <https://r-nimble.org/download>):


``` r
library(nimble)
```

Let’s return to our running example on coypu survival. First step: define the binomial likelihood and a uniform prior on the survival probability $\theta$ using the `nimbleCode()` function:


``` r
model <- nimbleCode({
  # likelihood
  y ~ dbinom(theta, n)
  # prior
  theta ~ dbeta(1, 1) # or dunif(0,1)
})
```

We can check that the `model` object indeed contains this code:


``` r
model
#> {
#>     y ~ dbinom(theta, n)
#>     theta ~ dbeta(1, 1)
#> }
```

In the code, `y` and `n` are known, and only $\theta$ needs to be estimated. The line `y ~ dbinom(theta, n)` indicates that the number of survivors follows a binomial distribution. The prior is a beta distribution with parameters 1 and 1 (`dbeta()`), i.e. a uniform distribution between 0 and 1 (`dunif()`). Standard distributions are available in `NIMBLE` (`dnorm`, `dpois`, `dmultinom`, etc.). Note that the order of the lines does not matter: `NIMBLE` uses a declarative language (you specify *what*, not *how*).

In a second step, we enter the data in a list:


``` r
dat <- list(n = 57, y = 19)
```

`NIMBLE` distinguishes data (known values on the left of `~`) from constants (e.g. loop indices). Declaring some values as constants can improve computational efficiency, although this is not always intuitive. Fortunately, `NIMBLE` largely handles this automatically and may suggest moving some objects to constants if it improves performance. We ignore this distinction here, but we will use it later in Chapter \@ref(glms).

The third step is to tell `NIMBLE` which parameters you want to monitor. Here, we are interested in the survival probability $\theta$:


``` r
par <- c("theta")
```

In general, your model contains many quantities, some of which are not very informative and do not need to be monitored. Having full control over what is tracked is therefore very useful.

The fourth step consists in specifying initial values for all model parameters. At a minimum, you must provide initial values for all quantities that appear only on the left side of `~` in your code and are not supplied as data.

To ensure that the MCMC algorithm properly explores the posterior distribution, we run multiple chains with different initial values. You can specify initial values for each chain (here three chains) in a list, which is itself placed inside another list:


``` r
init1 <- list(theta = 0.1)
init2 <- list(theta = 0.5)
init3 <- list(theta = 0.9)
inits <- list(init1, init2, init3)
inits
#> [[1]]
#> [[1]]$theta
#> [1] 0.1
#> 
#> 
#> [[2]]
#> [[2]]$theta
#> [1] 0.5
#> 
#> 
#> [[3]]
#> [[3]]$theta
#> [1] 0.9
```

Alternatively, you can write an `R` function that generates random initial values:


``` r
inits <- function() list(theta = runif(1,0,1))
inits()
#> $theta
#> [1] 0.3109711
```

I prefer using functions because the code is more compact and automatically adapts to the number of chains. If you use a function to generate initial values, it is always good practice to set a random seed beforehand so that you can reproduce the results:


``` r
seed <- 666
set.seed(seed)
```

Fifth and final step: you need to tell `NIMBLE` the number of chains (`n.chains`), the burn-in length (`n.burnin`), and the total number of iterations (`n.iter`):


``` r
n.iter <- 2000
n.burnin <- 300
n.chains <- 3
```

In `NIMBLE`, you specify the total number of iterations, so the number of posterior samples per chain will be equal to `n.iter - n.burnin`.

As a side note, to determine the length of the warm-up period (burn-in), you can run `NIMBLE` with `n.burnin <- 0` for a few hundred or thousand iterations and inspect the parameter trace to decide how many iterations are needed to reach convergence.  

`NIMBLE` also allows you to discard samples after the burn-in phase, which is called thinning. By default, `thinning = 1` (no samples are removed), meaning that all simulations are used to summarize the posterior distributions.

We now have all the ingredients to run our model, i.e. to generate samples from the posterior distribution of the parameters via MCMC simulations. We use the `nimbleMCMC()` function for this:


``` r
mcmc.output <- nimbleMCMC(code = model, # model
                          data = dat, # data
                          inits = inits, # initial values
                          monitors = par, # parameters to monitor
                          niter = n.iter, # total number of iterations
                          nburnin = n.burnin, # burn-in iterations
                          nchains = n.chains) # number of chains
#> |-------------|-------------|-------------|-------------|
#> |-------------------------------------------------------|
#> |-------------|-------------|-------------|-------------|
#> |-------------------------------------------------------|
#> |-------------|-------------|-------------|-------------|
#> |-------------------------------------------------------|
```

`NIMBLE` performs several internal steps that we will not detail here. The `nimbleMCMC()` function accepts other useful arguments. For example, `setSeed` lets you fix the random seed inside the MCMC call, ensuring you obtain exactly the same chains at each run—very useful for reproducibility and debugging. You can also request a summary of the output with `summary = TRUE`, or retrieve MCMC samples in the `coda::mcmc()` format with `samplesAsCodaMCMC = TRUE`. Finally, you can remove the progress bar with `progressBar = FALSE` if you find it too depressing during long simulations. See `?nimbleMCMC` for details.

Let’s take a look at the results, starting by examining what the `mcmc.output` object contains:


``` r
str(mcmc.output)
#> List of 3
#>  $ chain1: num [1:1700, 1] 0.407 0.201 0.451 0.273 0.254 ...
#>   ..- attr(*, "dimnames")=List of 2
#>   .. ..$ : NULL
#>   .. ..$ : chr "theta"
#>  $ chain2: num [1:1700, 1] 0.507 0.382 0.256 0.365 0.177 ...
#>   ..- attr(*, "dimnames")=List of 2
#>   .. ..$ : NULL
#>   .. ..$ : chr "theta"
#>  $ chain3: num [1:1700, 1] 0.317 0.244 0.317 0.362 0.357 ...
#>   ..- attr(*, "dimnames")=List of 2
#>   .. ..$ : NULL
#>   .. ..$ : chr "theta"
```

The `R` object `mcmc.output` is a list with three elements, one for each MCMC chain. Let’s look, for example, at the first chain:


``` r
dim(mcmc.output$chain1)
#> [1] 1700    1
head(mcmc.output$chain1)
#>          theta
#> [1,] 0.4070527
#> [2,] 0.2005720
#> [3,] 0.4513129
#> [4,] 0.2725412
#> [5,] 0.2539956
#> [6,] 0.4019970
```

Each element of the list is a matrix. The rows correspond to the 1700 samples from the posterior distribution of $\theta$ (which corresponds to `n.iter - n.burnin` iterations). The columns represent the parameters we monitor, here `theta`.

From there, we can compute the posterior mean of $\theta$:


``` r
mean(mcmc.output$chain1[,"theta"])
#> [1] 0.3391349
```

And the 95% credible interval:


``` r
quantile(mcmc.output$chain1[,"theta"], probs = c(2.5, 97.5)/100)
#>      2.5%     97.5% 
#> 0.2308179 0.4541410
```

Let us now visualize the posterior distribution of $\theta$ as a histogram:


``` r
mcmc.output$chain1[,"theta"] %>%
  as_tibble() %>%
  ggplot() +
  geom_histogram(aes(x = value), color = "white") +
  labs(x = "Survival probability")
```

<div class="figure" style="text-align: center">
<img src="03-implementation_files/figure-html/posterior-theta-1.png" alt="Histogram of the posterior distribution of the survival probability (\(\theta\))." width="90%" />
<p class="caption">(\#fig:posterior-theta)Histogram of the posterior distribution of the survival probability (\(\theta\)).</p>
</div>

There are more convenient ways to perform these Bayesian inferences. We will use the `R` package `MCMCvis` to summarize and visualize MCMC output, but you can also use `ggmcmc`, `bayesplot`, or `basicMCMCplots`.

Let’s load `MCMCvis`:


``` r
library(MCMCvis)
```

To obtain the most common numerical summaries, we use `MCMCsummary()`:


``` r
MCMCsummary(object = mcmc.output, round = 2)
#>       mean   sd 2.5%  50% 97.5% Rhat n.eff
#> theta 0.34 0.06 0.22 0.34  0.46    1  4831
```

We can also draw a caterpillar plot with `MCMCplot()` to visualize posterior distributions:


``` r
MCMCplot(object = mcmc.output, params = 'theta')
```

<div class="figure" style="text-align: center">
<img src="03-implementation_files/figure-html/caterpilla-theta-1.png" alt="Caterpillar plot of the posterior distribution of the survival probability (\(\theta\))." width="90%" />
<p class="caption">(\#fig:caterpilla-theta)Caterpillar plot of the posterior distribution of the survival probability (\(\theta\)).</p>
</div>

The point represents the posterior median, the thick bar the 50% credible interval, and the thin bar the 95% credible interval.

We can plot the MCMC chain (trace plot) and the associated posterior density with `MCMCtrace()`:


``` r
MCMCtrace(object = mcmc.output,
          pdf = FALSE,
          ind = TRUE,
          params = "theta")
```

<div class="figure" style="text-align: center">
<img src="03-implementation_files/figure-html/trace-theta-1.png" alt="Trace plot and posterior density of the survival probability (\(\theta\))." width="90%" />
<p class="caption">(\#fig:trace-theta)Trace plot and posterior density of the survival probability (\(\theta\)).</p>
</div>

These plots are used to assess chain convergence and to detect potential estimation issues (see Chapter \@ref(mcmc)). We can also add the diagnostics discussed earlier:


``` r
MCMCtrace(object = mcmc.output,
          pdf = FALSE,
          ind = TRUE,
          Rhat = TRUE,
          n.eff = TRUE,
          params = "theta")
```

<div class="figure" style="text-align: center">
<img src="03-implementation_files/figure-html/trace-theta2-1.png" alt="Trace plot and posterior density of the survival probability (\(\theta\)) with convergence diagnostics." width="90%" />
<p class="caption">(\#fig:trace-theta2)Trace plot and posterior density of the survival probability (\(\theta\)) with convergence diagnostics.</p>
</div>

A major advantage of MCMC methods is that they provide the posterior distribution of any function of the parameters by applying that function to draws from the posterior distributions of those parameters. For example, suppose we want to compute the life expectancy of coypus, given by $\lambda = -1/\log(\theta)$. 

In our example, we simply combine the `theta` samples from the three chains:


``` r
theta_samples <- c(mcmc.output$chain1[,"theta"],
                   mcmc.output$chain2[,"theta"],
                   mcmc.output$chain3[,"theta"])
```

Then compute the corresponding life expectancy:


``` r
lambda <- -1/log(theta_samples)
```

We thus obtain 5100 simulated values from the posterior distribution of $\lambda$, whose first values are:


``` r
head(lambda)
#> [1] 1.1125791 0.6224394 1.2569220 0.7692513 0.7296935 1.0973206
```

We can then extract the usual summaries:


``` r
mean(lambda)
#> [1] 0.9372371
quantile(lambda, probs = c(2.5, 97.5)/100)
#>      2.5%     97.5% 
#> 0.6691676 1.2999116
```

Life expectancy is approximately one year. We can also visualize the posterior distribution of life expectancy:


``` r
lambda %>%
  as_tibble() %>%
  ggplot() +
  geom_histogram(aes(x = value), color = "white") +
  labs(x = "Life expectancy")
```

<div class="figure" style="text-align: center">
<img src="03-implementation_files/figure-html/hist-life-nimble-1.png" alt="Histogram of the posterior distribution of life expectancy." width="90%" />
<p class="caption">(\#fig:hist-life-nimble)Histogram of the posterior distribution of life expectancy.</p>
</div>

We could also compute life expectancy by inserting it directly into the NIMBLE model with a line `lambda <- -1/log(theta)` and adding `lambda` to the monitored outputs. The approach presented here is particularly useful with large models and/or large datasets, because it reduces memory usage.

Now you can get started. For convenience, the steps above are summarized below. The workflow provided by `nimbleMCMC()` allows you to build models and perform Bayesian inference:


``` r
model <- nimbleCode({
  y ~ dbinom(theta, n)
  theta ~ dbeta(1, 1)
  lambda <- -1/log(theta)
})
dat <- list(n = 57, y = 19)
par <- c("theta", "lambda")
inits <- function() list(theta = runif(1,0,1))
n.iter <- 5000
n.burnin <- 1000
n.chains <- 3
mcmc.output <- nimbleMCMC(code = model,
                          data = dat,
                          inits = inits,
                          monitors = par,
                          niter = n.iter,
                          nburnin = n.burnin,
                          nchains = n.chains)
MCMCsummary(object = mcmc.output, round = 2)
MCMCplot(object = mcmc.output)
MCMCtrace(object = mcmc.output, pdf = FALSE, ind = TRUE)
```

In this section, we introduced the bare minimum to get started with `NIMBLE`. But `NIMBLE` is much more than a simple MCMC engine: it is a programming environment that gives you full control over model construction and parameter estimation. You can write your own functions and distributions, choose MCMC methods yourself, or even code your own algorithms. See the manual <https://r-nimble.org/html_manual/cha-welcome-nimble.html> for more details.

## `brms`

`brms` stands for **B**ayesian **R**egression **M**odels using **S**tan. This package makes it possible to formulate and estimate regression models (see the next section and Chapters \@ref(lms) and \@ref(glms)) in an intuitive way thanks to a syntax close to that of the `lme4` package (the `R` reference for mixed models), while relying on `Stan`, a reference software in Bayesian statistics. The package is under constant development; see <https://paul-buerkner.github.io/brms/>. You can get help via <https://discourse.mc-stan.org/>. 

To use `brms`, we start by preparing the data:

``` r
dat <- data.frame(y = 19, n = 57)
```

Without forgetting to load `brms`:

``` r
library(brms)
```

The likelihood is binomial in our running example. In `brms`, we can express this simply:



``` r
bayes.brms <- brm(
  y | trials(n) ~ 1, # the number of successes is a function of an intercept
  family = binomial("logit"), # binomial family with logit link function
  data = dat, # data used
  chains = 3, # number of MCMC chains
  iter = 2000, # total number of iterations per chain
  warmup = 300, # number of burn-in iterations
  thin = 1 # no thinning (each iteration is kept)
)
```

The syntax is relatively simple but requires a few explanations. The argument `y | trials(n) ~ 1` makes it possible to specify a model in which we have $y$ successes among $n$ trials, and we estimate only an intercept, the `1` after `~`. Why an intercept here? Why not directly the survival $\theta$? Because we use `family = binomial("logit")` on the next line to specify to `brms` that the response variable follows a binomial distribution. In other words, we have a generalized linear model (see Chapter \@ref(glms)) with $\text{logit}(\theta) = \beta$ and we estimate $\beta$, the intercept. The arguments `iter = 2000`, `warmup = 300`, and `chains = 3` tell `brms` to use 300 iterations for adaptation (burn-in), and the following 1700 for inference, with 3 chains. 

Let’s take a look at the results: 

``` r
summary(bayes.brms)
#>  Family: binomial 
#>   Links: mu = logit 
#> Formula: y | trials(n) ~ 1 
#>    Data: dat (Number of observations: 1) 
#>   Draws: 3 chains, each with iter = 2000; warmup = 300; thin = 1;
#>          total post-warmup draws = 5100
#> 
#> Regression Coefficients:
#>           Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
#> Intercept    -0.70      0.28    -1.28    -0.17 1.00     1732     2305
#> 
#> Draws were sampled using sampling(NUTS). For each parameter, Bulk_ESS
#> and Tail_ESS are effective sample size measures, and Rhat is the potential
#> scale reduction factor on split chains (at convergence, Rhat = 1).
```



This command displays a summary table of posterior estimates for each parameter of the model. We find there:

- `Estimate` is the posterior mean.
- `Est.Error` is the standard deviation of the posterior distribution.
- `l-95% CI` and `u-95% CI` are the bounds of the 95% credible interval.
- The convergence diagnostic `Rhat`.
- `Bulk_ESS` is the effective sample size (`Tail_ESS` is another measure of effective sample size that we will not use here). 

The posterior mean is -0.7 far from the proportion of coypus that survived the winter ($19/57 \approx 0.33$). As always in `R` and in the implementation of generalized linear models (see Chapter \@ref(glms)), parameter estimates are given on the scale of the link function. Here, the estimated intercept is expressed on the logit scale. To convert it to a survival probability (between 0 and 1), we first extract the values generated in the posterior distribution of the intercept $\beta$ with the function `brms::as_draws_matrix()`:

``` r
draws_fit <- as_draws_matrix(bayes.brms)
```

Then we apply the inverse logistic function `plogis()` to each of these values to obtain a whole bunch of simulated values from the posterior distribution of survival $\theta$:

``` r
beta <- draws_fit[,'Intercept'] # selects the intercept column
theta <- plogis(beta)  # logit -> [0,1] conversion
```

We thus obtain a direct estimate of the posterior mean of the survival probability, along with its 95% credible interval:

``` r
mean(theta)
#> [1] 0.3354256
quantile(theta, probas = c(2.5,97.5)/100)
#>        0%       25%       50%       75%      100% 
#> 0.1555931 0.2932298 0.3331265 0.3770575 0.5527164
```

Or more directly with the function `posterior::summarise_draws()`:

``` r
summarise_draws(theta)
#> # A tibble: 1 × 10
#>   variable   mean median     sd    mad    q5   q95  rhat ess_bulk ess_tail
#>   <chr>     <dbl>  <dbl>  <dbl>  <dbl> <dbl> <dbl> <dbl>    <dbl>    <dbl>
#> 1 Intercept 0.335  0.333 0.0617 0.0619 0.235 0.440  1.00    1732.    2305.
```

To visualize the posterior distribution of survival probability, we just need to use (Figure \@ref(fig:hist-surviebrms)):

``` r
draws_fit %>%
  ggplot(aes(x = theta)) +
  geom_histogram(color = "white", fill = "steelblue", bins = 30) +
  labs(x = "Survival probability", y = "Frequency")
```

<div class="figure" style="text-align: center">
<img src="03-implementation_files/figure-html/hist-surviebrms-1.png" alt="Histogram of the posterior distribution of the survival probability (\(\theta\))." width="90%" />
<p class="caption">(\#fig:hist-surviebrms)Histogram of the posterior distribution of the survival probability (\(\theta\)).</p>
</div>

In `brms`, we can assess the convergence of the MCMC chains (Figure \@ref(fig:trace-surviebrms)):

``` r
plot(bayes.brms)
```

<div class="figure" style="text-align: center">
<img src="03-implementation_files/figure-html/trace-surviebrms-1.png" alt="Histogram of the posterior distribution and trace plot of the survival probability on the logit scale (b). In the histogram, the x-axis represents the possible values of the intercept (logit scale) and the y-axis the frequency of the simulated values. In the trace plot, the x-axis corresponds to the MCMC iteration number and the y-axis to the simulated values of the intercept (logit scale)." width="90%" />
<p class="caption">(\#fig:trace-surviebrms)Histogram of the posterior distribution and trace plot of the survival probability on the logit scale (b). In the histogram, the x-axis represents the possible values of the intercept (logit scale) and the y-axis the frequency of the simulated values. In the trace plot, the x-axis corresponds to the MCMC iteration number and the y-axis to the simulated values of the intercept (logit scale).</p>
</div>

This graph displays trace plots (right) as well as posterior densities (left). 

As a side note, to determine the length of the warm-up period (burn-in), it is enough to run `brms` with `warmup = 0` for a few hundred or thousand iterations and inspect the parameter trace to decide the number of iterations needed to reach convergence.  

A major advantage of MCMC methods is that they allow obtaining the posterior distribution of any function of the parameters by applying this function to the values drawn from the posterior distributions of these parameters. Note that here we estimate the intercept $\beta$ and we have therefore already used this idea to obtain the posterior distribution of the survival probability by applying the inverse logit function. As another example, suppose I would like to compute the life expectancy of coypus, which is given by $\lambda = -1/\log(\theta)$:

``` r
beta <- draws_fit[,'Intercept'] # selects the intercept column
theta <- plogis(beta)  # logit -> [0,1] conversion
lambda <- -1 / log(theta) # transforms survival into life expectancy
summarize_draws(lambda) # summary of draws: mean, median, intervals
#> # A tibble: 1 × 10
#>   variable   mean median    sd   mad    q5   q95  rhat ess_bulk ess_tail
#>   <chr>     <dbl>  <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>    <dbl>    <dbl>
#> 1 Intercept 0.928  0.910 0.161 0.153 0.691  1.22  1.00    1732.    2305.
```

Life expectancy is approximately one year. We can also visualize the posterior distribution of life expectancy (Figure \@ref(fig:hist-life)):

``` r
lambda %>%
  as_tibble() %>%
  ggplot() +
  geom_histogram(aes(x = Intercept), color = "white") +
  labs(x = "Life expectancy")
```

<div class="figure" style="text-align: center">
<img src="03-implementation_files/figure-html/hist-life-1.png" alt="Histogram of the posterior distribution of life expectancy. The x-axis represents the different possible values of life expectancy. The vertical axis indicates the number of simulated draws (Count) for each value." width="90%" />
<p class="caption">(\#fig:hist-life)Histogram of the posterior distribution of life expectancy. The x-axis represents the different possible values of life expectancy. The vertical axis indicates the number of simulated draws (Count) for each value.</p>
</div>

There are a whole bunch of parameters that are set by default in `brms`; it is important to be aware of them. This concerns priors in particular. In `brms`, default priors are often non-informative or weakly informative, but it is always good to examine them explicitly. The following command displays a summary of the priors used in an already fitted model:

``` r
prior_summary(bayes.brms)
#> Intercept ~ student_t(3, 0, 2.5)
```

The `brms` package uses as a weakly informative prior a Student distribution with 3 degrees of freedom, centered at 0, with a standard deviation of 2.5. The 3 degrees of freedom give a distribution with heavier tails than a normal, which provides some robustness to extreme values. The center at 0 reflects an absence of strong prior on the value of the intercept. The width 2.5 allows reasonably wide variation of the intercept without being completely non-informative.

In some cases, it is relevant to define your own prior, for example to reflect knowledge from the literature or to further constrain estimation (informative prior). Here, we propose a normal prior centered at 0 with a standard deviation of 1.5 on the intercept; we will come back to this in Chapter \@ref(prior):

``` r
nlprior <- prior(normal(0, 1.5), class = "Intercept")
```

We can then use it in the model specification:



``` r
bayes.brms <- brm(y | trials(n) ~ 1,
                  family = binomial("logit"),
                  data = dat,
                  prior = nlprior, # our own priors
                  chains = 3,
                  iter = 2000,
                  warmup = 300,
                  thin = 1)
```

You can check that the results are close to those obtained with the default prior:

``` r
summary(bayes.brms)
#>  Family: binomial 
#>   Links: mu = logit 
#> Formula: y | trials(n) ~ 1 
#>    Data: dat (Number of observations: 1) 
#>   Draws: 3 chains, each with iter = 2000; warmup = 300; thin = 1;
#>          total post-warmup draws = 5100
#> 
#> Regression Coefficients:
#>           Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
#> Intercept    -0.69      0.27    -1.24    -0.18 1.00     1664     2306
#> 
#> Draws were sampled using sampling(NUTS). For each parameter, Bulk_ESS
#> and Tail_ESS are effective sample size measures, and Rhat is the potential
#> scale reduction factor on split chains (at convergence, Rhat = 1).
```

## Summary

- `NIMBLE` makes it possible to model both simple situations and complex models, with great flexibility.

- Its syntax is based on `R`, which makes it easier to get started if you know the language.

- It offers full control over the model and the algorithms, but assumes you are comfortable with programming.

- Conversely, `brms` makes it possible to take advantage of MCMC methods without having to write the model yourself (the likelihood in particular).

- Its syntax is simple and close to that of `lme4`, which makes it particularly suitable for generalized linear models (mixed or not; see Chapter \@ref(glms)).

- In return, `brms` relies on pre-programmed components (model families, etc.), and it is important to pay attention to default choices, especially regarding prior distributions.

- This chapter thus offers a first concrete approach to implementing Bayesian models, before moving on to richer models, such as mixed models.
