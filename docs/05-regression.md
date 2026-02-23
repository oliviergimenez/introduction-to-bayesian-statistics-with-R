
# Regression {#lms}

## Introduction 

This chapter presents the application of Bayesian statistics to linear regression. We will use an example that allows us to go a bit further than our running example on survival. This will be an opportunity to discuss how and why to use a model to simulate data. We will also illustrate model comparison and validation. We will use `NIMBLE` and `brms` and compare with the frequentist approach.

## Linear regression

### The model

To change things a bit, I suggest using `NIMBLE` and `brms` on an example different from survival estimation. Let us focus on linear regression.  

Let us start by laying out the foundations of our linear model. We have $n$ measurements of a response variable $y_i$ with $i$ ranging from 1 to $n$. Think for example of the mass (in kilograms) of our coypus in the running example. We associate each measurement with an explanatory variable $x_i$, for example the average outdoor temperature in winter (in degrees Celsius) for our coypus. We want to study the effect of temperature on mass. The simplest assumption is a linear relationship between the two, so we use a linear regression model. The model includes an intercept $\beta_0$, and a slope $\beta_1$ that describes the effect of $x_i$ on $y_i$, or of temperature on coypu mass. We also need a parameter to describe residual variability represented by a variance parameter $\sigma^2$, which captures the part of variation in the $y_i$ not explained by the $x_i$. You have probably already encountered this model in the form: $y_i = \beta_0 + \beta_1 x_i + \varepsilon_i$ where the errors $\varepsilon_i$ are assumed independent and normally distributed with mean 0 and variance $\sigma^2$. 

The intercept $\beta_0$ gives us the mass when the temperature is 0 degrees ($x_i = 0$). The parameter $\beta_1$ tells us the change in the response variable for a one‑unit increase (here 1 degree Celsius) in the explanatory variable (hence the term “slope” for this parameter). In general, it is (strongly) recommended to center (subtract the mean) and scale (divide by the standard deviation) the values of the explanatory variable for numerical and interpretational reasons. Numerical first, because it allows algorithms, whether frequentist or Bayesian, not to get lost in corners of the parameter space. Interpretation next, because the intercept $\beta_0$ is then interpreted as the value of the response variable for an average value of the explanatory variable. 

In this section, rather than analyzing “real” data, we will, from the parameters $\beta_0$, $\beta_1$ and $\sigma$, simulate artificial data, as if they came from a real underlying process. 

### Simulating data

What do I mean by simulating data? Data analysis and data simulation are two sides of the same model. In analysis, we use the data to estimate the parameters of a model. In simulation, we fix the parameters and use the model to generate data. One reason to use simulations is that this exercise forces us to really understand the model; if I cannot simulate data from a model, it means I have not fully understood how it works. There are many other good reasons to use simulations. Since the truth (the parameters and the model) is known, we can check that the model is correctly coded. We can evaluate bias and precision of our parameter estimates, assess the effects of violating model assumptions, plan a data collection protocol, or evaluate the power of a statistical test. In short, it is a very useful technique to have in your toolbox! 

Let us return to our example. To simulate data according to the linear regression model, we start by fixing our parameters: $\beta_0 = 0.1$, $\beta_1 = 1$ and $\sigma^2 = 0.5$ : 

``` r
beta0 <- 0.1 # true value of the intercept
beta1 <- 1 # true value of the coefficient of x
sigma <- 0.5 # standard deviation of the errors
```

Then we simulate $n = 100$ values $x_i$ of our explanatory variable from a normal distribution with mean 0 and standard deviation 1, that is $N(0,1)$ :

``` r
set.seed(666) # to make the simulation reproducible
n <- 100 # number of observations
x <- rnorm(n = n, mean = 0, sd = 1) # covariate x simulated from a standard normal distribution
```

Finally, we simulate the values of the response variable by adding a normal error `epsilon` to the linear relationship `beta0 + beta1 * x` : 

``` r
epsilon <- rnorm(n, mean = 0, sd = sigma) # generate normal errors
y <- beta0 + beta1 * x + epsilon # add errors to the linear relationship
data <- data.frame(y = y, x = x)
```

Figure \@ref(fig:donnees-simulees) below shows the simulated data, as well as the regression line corresponding to the model used to generate them : 
<div class="figure" style="text-align: center">
<img src="05-regression_files/figure-html/donnees-simulees-1.png" alt="Simulated data (n = 100) according to the model \(y_i = \beta_0 + \beta_1 x_i + \varepsilon_i\), with \(\beta_0 = 0.1\), \(\beta_1 = 1\) and \(\sigma = 1\). The red line corresponds to the regression line." width="90%" />
<p class="caption">(\#fig:donnees-simulees)Simulated data (n = 100) according to the model \(y_i = \beta_0 + \beta_1 x_i + \varepsilon_i\), with \(\beta_0 = 0.1\), \(\beta_1 = 1\) and \(\sigma = 1\). The red line corresponds to the regression line.</p>
</div>

### Fitting with `brms`



In this section, we use `brms` to fit the linear regression model to the data we have just generated. If everything goes well, the estimated parameters should be close to the values used to generate the data. I will go relatively quickly here since we covered the different steps in Chapter \@ref(software). The syntax is very close to what we would use to fit the model by maximum likelihood with the `lm()` function in `R`:


``` r
lm.brms <- brm(y ~ x, # formula: y as a function of x
               data = data, # dataset
               family = gaussian) # normal distribution
```

Let's take a look at the numerical summaries and the convergence diagnostics:

``` r
summary(lm.brms)
#>  Family: gaussian 
#>   Links: mu = identity; sigma = identity 
#> Formula: y ~ x 
#>    Data: data (Number of observations: 100) 
#>   Draws: 4 chains, each with iter = 2000; warmup = 1000; thin = 1;
#>          total post-warmup draws = 4000
#> 
#> Regression Coefficients:
#>           Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
#> Intercept     0.06      0.06    -0.05     0.17 1.00     4366     3028
#> x             1.10      0.06     0.99     1.21 1.00     4188     3147
#> 
#> Further Distributional Parameters:
#>       Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
#> sigma     0.57      0.04     0.49     0.65 1.00     4090     3050
#> 
#> Draws were sampled using sampling(NUTS). For each parameter, Bulk_ESS
#> and Tail_ESS are effective sample size measures, and Rhat is the potential
#> scale reduction factor on split chains (at convergence, Rhat = 1).
```

By default, `brms` used four chains that each ran for 2000 iterations with 1000 iterations used as burn-in, for a total of 4000 iterations for posterior inference. In the output, `Intercept`, `x` and `sigma` correspond respectively to the parameters $\beta_0$, $\beta_1$ and $\sigma$ of the model. The \( \hat{R} \) for the 3 parameters is 1, and the effective sample sizes are satisfactory. The credible intervals contain the true parameter value used to simulate the data.

We check that the mixing is good (Figure \@ref(fig:fig-posterior-regression)):

``` r
plot(lm.brms)
```

<div class="figure" style="text-align: center">
<img src="05-regression_files/figure-html/fig-posterior-regression-1.png" alt="Histograms of the posterior distributions (left column) and traces (right column) of the linear regression parameters. In the histograms, the x-axis represents the possible values of the estimated parameter (intercept, slope, or standard deviation) and the y-axis corresponds to their frequency in the posterior sample. In the trace plots, the x-axis indicates the MCMC iteration number, while the y-axis represents the simulated value of the parameter at each iteration." width="90%" />
<p class="caption">(\#fig:fig-posterior-regression)Histograms of the posterior distributions (left column) and traces (right column) of the linear regression parameters. In the histograms, the x-axis represents the possible values of the estimated parameter (intercept, slope, or standard deviation) and the y-axis corresponds to their frequency in the posterior sample. In the trace plots, the x-axis indicates the MCMC iteration number, while the y-axis represents the simulated value of the parameter at each iteration.</p>
</div>

### Weakly informative priors {#weakly-informative-priors}

Rather than using the default priors in `brms`, let's choose other priors. We will use weakly informative priors, and more specifically a normal with mean 0 and standard deviation 1.5, or $N(0, 1.5)$, for the regression parameters $\beta_0$ and $\beta_1$. We already discussed weakly informative priors in Chapter \@ref(prior). The idea is close to that of vague or non-informative priors, in the sense that we try, through weakly informative priors, to reflect the fact that we do not really have information on the model parameters. The difference is that non-informative priors can induce aberrant values as we saw in Chapter \@ref(prior). This is still the case here. Take for example $N(0, 100)$ for the parameters of the linear relationship that links the mass of coypus to temperature, and simulate a whole bunch of values from these priors, then form the linear relationship:

``` r

# number of lines to simulate
n_lines <- 100

# draws of intercepts and slopes from the priors
intercepts <- rnorm(n_lines, mean = 0, sd = 100)
slopes <- rnorm(n_lines, mean = 0, sd = 100)

# create a data frame
lines_df <- data.frame()
for (i in 1:n_lines) {
  y_vals <- intercepts[i] + slopes[i] * x
  temp_df <- data.frame(x = x, y = y_vals, line = as.factor(i))
  lines_df <- rbind(lines_df, temp_df)
}

# plot with ggplot2
ggplot(lines_df, aes(x = x, y = y, group = line)) +
  geom_line(alpha = 0.3) +
  theme_minimal() +
  labs(x = "x", y = "y")
```

<div class="figure" style="text-align: center">
<img src="05-regression_files/figure-html/fig-prior-regression-vague-1.png" alt="Simulation of regression lines drawn from the prior distributions. Each line corresponds to a draw of the parameters: intercept and slope ~ N(0, 100)." width="90%" />
<p class="caption">(\#fig:fig-prior-regression-vague)Simulation of regression lines drawn from the prior distributions. Each line corresponds to a draw of the parameters: intercept and slope ~ N(0, 100).</p>
</div>

In Figure \@ref(fig:fig-prior-regression-vague), we see that we obtain aberrant values for the $y_i$, with coypus weighing more than 400 kilograms, and (very) negative values for the mass. We have just done a “prior predictive check”, as in Chapter \@ref(prior). In Figure \@ref(fig:fig-prior-regression), we do the same thing with our weakly informative prior $N(0, 1.5)$:
<div class="figure" style="text-align: center">
<img src="05-regression_files/figure-html/fig-prior-regression-1.png" alt="Simulation of regression lines drawn from the prior distributions. Each line corresponds to a draw of the parameters: intercept and slope ~ N(0, 1.5)." width="90%" />
<p class="caption">(\#fig:fig-prior-regression)Simulation of regression lines drawn from the prior distributions. Each line corresponds to a draw of the parameters: intercept and slope ~ N(0, 1.5).</p>
</div>

We obtain more reasonable values for the mass of coypus, which rarely exceeds 10 kilograms. We still have negative values, but smaller ones, and the MCMC algorithm should cope. There is also a numerical advantage to using weakly informative priors: they help MCMC methods not to get lost in the space of all possible values for the parameters to be estimated, and allow them to focus on realistic values of these parameters. By doing this, you may have the impression that we are using the data to construct the priors, whereas we said that the prior should reflect the information available before seeing the data. This is an opportunity to clarify this point a bit. The important thing is above all that the prior represents information independent of the data that are used in the likelihood.

So far we have focused on the regression parameters, the intercept $\beta_0$ and the slope $\beta_1$. But what about the standard deviation, $\sigma$? This parameter is just as important: it reflects how much the observations deviate from the average trend described by the regression line.

One option often considered is to assign it a uniform distribution, for example $\sigma \sim U(0, B)$, with a natural lower bound (0, since $\sigma$ is always positive), but an upper bound $B$ that is difficult to choose. What maximum value should one give to a standard deviation? In some cases, an apparently reasonable value can turn out to be too wide. For example, if we model human heights and set $\sigma \sim U(0, 50)$ (in cm), this amounts to assuming that 95% of heights are spread over a 100 cm range around the mean—which is very unlikely.

A more flexible and more realistic alternative is to use an exponential distribution $\sigma \sim \exp(\lambda)$ where $\lambda > 0$ is a rate parameter. This distribution is defined only for positive values, which is consistent with the nature of $\sigma$, and it favors small values of the standard deviation while leaving the possibility for $\sigma$ to be larger if the data justify it.

By default, one often takes $\lambda = 1$. With $\lambda = 1$, the mean and the standard deviation of this distribution are both equal to $1$, which induces a modest but non-restrictive prior (Figure \@ref(fig:fig-prior-sigma)).

<div class="figure" style="text-align: center">
<img src="05-regression_files/figure-html/fig-prior-sigma-1.png" alt="Comparison between two prior distributions for the standard deviation \(\sigma\): a uniform distribution \(\text{U}(0,5)\), which gives the same density between 0 and 5, and an exponential distribution \(\text{Exp}(1)\), which favors small values while retaining a heavier tail." width="90%" />
<p class="caption">(\#fig:fig-prior-sigma)Comparison between two prior distributions for the standard deviation \(\sigma\): a uniform distribution \(\text{U}(0,5)\), which gives the same density between 0 and 5, and an exponential distribution \(\text{Exp}(1)\), which favors small values while retaining a heavier tail.</p>
</div>

We can formalize this model as follows:
\begin{align}
y_i &\sim \text{Normal}(\mu_i, \sigma^2) &\text{[likelihood]}\\
\mu_i &= \beta_0 + \beta_1 \; x_i &\text{[linear relationship]}\\
\beta_0, \beta_1 &\sim \text{Normal}(0, 1.5) &\text{[prior on parameters]} \\
\sigma &\sim \text{Exp}(1) &\text{[prior on parameters]} \\
\end{align}

Let us specify these priors:

``` r
myprior <- c(
  prior(normal(0, 1.5), class = b), # prior on the coefficient of x
  prior(normal(0, 1.5), class = Intercept), # prior on the intercept
  prior(exponential(1), class = sigma) # prior on the standard deviation of the error
)
```

Then let's refit with `brms`:



``` r
lm.brms <- brm(y ~ x, 
               data = data, 
               family = gaussian, 
               prior = myprior)
```

We check that the numerical summaries obtained are close to those obtained with the default priors, and above all close to the values used to simulate the data:

``` r
summary(lm.brms)
#>  Family: gaussian 
#>   Links: mu = identity; sigma = identity 
#> Formula: y ~ x 
#>    Data: data (Number of observations: 100) 
#>   Draws: 4 chains, each with iter = 2000; warmup = 1000; thin = 1;
#>          total post-warmup draws = 4000
#> 
#> Regression Coefficients:
#>           Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
#> Intercept     0.06      0.06    -0.05     0.18 1.00     3562     2765
#> x             1.10      0.06     0.99     1.21 1.00     3870     2731
#> 
#> Further Distributional Parameters:
#>       Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
#> sigma     0.57      0.04     0.49     0.66 1.00     3540     2633
#> 
#> Draws were sampled using sampling(NUTS). For each parameter, Bulk_ESS
#> and Tail_ESS are effective sample size measures, and Rhat is the potential
#> scale reduction factor on split chains (at convergence, Rhat = 1).
```

Here, the two models give almost the same thing, which is not surprising because the data are informative enough for them to “take over from” the prior. The interest of weakly informative priors is not so much seen in this small example as in other situations: they avoid aberrant values, stabilize the MCMC computations, and remain useful when we have fewer data or more complex models.

### Fitting with `NIMBLE`



We start by writing the model: 

``` r
model <- nimbleCode({
  # priors
  beta0 ~ dnorm(0, sd = 1.5) # normal prior on intercept
  beta1 ~ dnorm(0, sd = 1.5) # normal prior on coefficient
  sigma ~ dexp(1) # exponential prior on standard deviation
  # likelihood
  for(i in 1:n) {
    y[i] ~ dnorm(beta0 + beta1 * x[i], sd = sigma) # equiv of yi = beta0 + beta1 * xi + epsiloni
  }
})
```

In this code block, we start by specifying priors on the three model parameters: a normal prior centered on 0 with standard deviation 1.5 for the intercept $\beta_0$ and for the slope $\beta_1$, as well as an exponential prior for the standard deviation $\sigma$ of the errors. The next part is a `for(i in 1:n)` loop that defines the likelihood. We specify the likelihood observation by observation, and `NIMBLE` automatically deduces the product of likelihoods over all individuals, which corresponds to the likelihood of the dataset. For each observation $i$, we have a normal distribution centered at `beta0 + beta1 * x[i]`, with standard deviation `sigma`. We recover the relationship $y_i = \beta_0 + \beta_1 x_i + \varepsilon_i$ where $\varepsilon_i \sim N(0,\sigma^2)$, which is strictly equivalent to $y_i \sim N(\beta_0 + \beta_1 x_i,\sigma^2)$.

The next steps are to put the data into a list, specify initial values, and indicate the parameters for which we want output:  

``` r
dat <- list(x = x, y = y, n = n) # data
inits <- list(list(beta0 = -0.5, beta1 = -0.5, sigma = 0.1), # inits chain 1
              list(beta0 = 0, beta1 = 0, sigma = 1), # inits chain 2
              list(beta0 = 0.5, beta1 = 0.5, sigma = 0.5)) # inits chain 3
par <- c("beta0", "beta1", "sigma")
```

We then have all the ingredients to run `NIMBLE`: 


``` r
lm.nimble <- nimbleMCMC(
  code = model,
  data = dat,
  inits = inits,
  monitors = par,
  niter = 2000,
  nburnin = 1000,
  nchains = 3
)
```



Let’s inspect the results: 

``` r
MCMCsummary(lm.nimble, round = 2)
#>       mean   sd  2.5%  50% 97.5% Rhat n.eff
#> beta0 0.06 0.06 -0.05 0.06  0.17 1.00  3000
#> beta1 1.10 0.06  0.99 1.10  1.21 1.00  3000
#> sigma 0.57 0.04  0.49 0.56  0.65 1.01   772
```

We obtain numerical summaries that are close to those obtained with `brms`, and close to the true parameter values used to simulate the data. 

For convergence, we can inspect the trace plots:

``` r
MCMCtrace(object = lm.nimble,
          pdf = FALSE,
          ind = TRUE,
          Rhat = TRUE,
          n.eff = TRUE)
```

<img src="05-regression_files/figure-html/unnamed-chunk-19-1.png" width="90%" style="display: block; margin: auto;" />

Everything looks good. Mixing is correct, and the convergence diagnostics are in the green. 

### Maximum likelihood fitting

Finally, we can compare with maximum likelihood fitting, obtained simply with the command `lm(y ~ x, data = data)`. Everything is in Figure \@ref(fig:comparaison-methodes):


<div class="figure" style="text-align: center">
<img src="05-regression_files/figure-html/comparaison-methodes-1.png" alt="Comparison of parameter estimates (intercept and slope) across methods (brms, lm, and NIMBLE). Points show posterior means for brms and NIMBLE, and the maximum likelihood estimate for lm. We also show 95% credible intervals (brms and NIMBLE) and the 95% confidence interval (lm). The dashed black line indicates the true value used to simulate the data." width="90%" />
<p class="caption">(\#fig:comparaison-methodes)Comparison of parameter estimates (intercept and slope) across methods (brms, lm, and NIMBLE). Points show posterior means for brms and NIMBLE, and the maximum likelihood estimate for lm. We also show 95% credible intervals (brms and NIMBLE) and the 95% confidence interval (lm). The dashed black line indicates the true value used to simulate the data.</p>
</div>

The posterior means obtained with `NIMBLE` and `brms` are close to the maximum likelihood estimates for the intercept and the slope, to a lesser extent. The credible intervals obtained with `NIMBLE` and `brms` and the confidence interval obtained by maximum likelihood all include the true parameter values used to simulate the data. Keep in mind that this is a single simulation; the exercise would need to be repeated many times to formally assess the distance between the true values and the parameter estimates (bias). 

## Model evaluation

The quality of a model fit to data is essential to assess how much confidence we can place in parameter estimates. Goodness-of-fit tests are well established in frequentist statistics, and many of them can also be used in simple Bayesian models. This is the case, for example, for residual analysis. 

In the case of linear regression, the model rests on several assumptions. These are the assumptions of independence, normality, linearity, and homoscedasticity ($\sigma$ does not vary with the explanatory variable). In general, we can evaluate the first two with context. For the other two, we can visualize the fit by overlaying the estimated regression line on the observed scatter plot. With the `brms` package, this gives Figure \@ref(fig:brms-fit-plot):

``` r
# extract values from posteriors
post <- as_draws_df(lm.brms)

# create grid of x values
grille_x <- tibble(x = seq(min(data$x), max(data$x), length.out = 100))

# for each x, simulate y values
pred <- post %>%
  select(b_Intercept, b_x) %>%
  expand_grid(grille_x) %>%
  mutate(y = b_Intercept + b_x * x) %>%
  group_by(x) %>%
  summarise(
    mean = mean(y),
    lower = quantile(y, 0.025),
    upper = quantile(y, 0.975),
    .groups = "drop"
  )

# extract post means
intercept <- summary(lm.brms)$fixed[1,1]
slope <- summary(lm.brms)$fixed[2,1]

# dataviz
ggplot(data, aes(x = x, y = y)) +
  geom_point(alpha = 0.6) +
  geom_ribbon(data = pred, aes(x = x, ymin = lower, ymax = upper), fill = "blue", alpha = 0.2, inherit.aes = FALSE) +
  geom_line(data = pred, aes(x = x, y = mean), color = "blue", size = 1.2) +
  labs(x = "x", y = "y") +
  coord_cartesian(xlim = range(grille_x$x)) +
  theme_minimal()
```

<div class="figure" style="text-align: center">
<img src="05-regression_files/figure-html/brms-fit-plot-1.png" alt="Linear model fit with brms. The blue line is the estimated regression, obtained by setting the intercept and slope to their posterior means, surrounded by its 95% credible interval." width="90%" />
<p class="caption">(\#fig:brms-fit-plot)Linear model fit with brms. The blue line is the estimated regression, obtained by setting the intercept and slope to their posterior means, surrounded by its 95% credible interval.</p>
</div>

With `NIMBLE`, this is Figure \@ref(fig:nimble-fit-plot): 

``` r
x <- data$x
y <- data$y

posterior <- rbind(lm.nimble$chain1, lm.nimble$chain2, lm.nimble$chain3)
beta0 <- posterior[,'beta0']
beta1 <- posterior[,'beta1']

x_seq <- seq(min(data$x), max(data$x), length.out = 100)

pred_matrix <- sapply(x_seq, function(xi) beta0 + beta1 * xi)

pred_df <- tibble(
  x = x_seq,
  y_mean = colMeans(pred_matrix),
  y_lower = apply(pred_matrix, 2, quantile, probs = 0.025),
  y_upper = apply(pred_matrix, 2, quantile, probs = 0.975)
)

true_df <- tibble(x = x_seq, y_true = 0.1 + 1 * x_seq)

ggplot() +
  geom_point(data = data, aes(x = x, y = y), alpha = 0.6) +
  geom_ribbon(data = pred_df, aes(x = x, ymin = y_lower, ymax = y_upper), fill = "blue", alpha = 0.2) +
  geom_line(data = pred_df, aes(x = x, y = y_mean), color = "blue", size = 1.2) +
 # geom_line(data = true_df, aes(x = x, y = y_true), color = "red", size = 1.2) +
  labs(x = "x", y = "y") +
  theme_minimal()
```

<div class="figure" style="text-align: center">
<img src="05-regression_files/figure-html/nimble-fit-plot-1.png" alt="Linear model fit with NIMBLE. The blue line is the estimated regression, obtained by setting the intercept and slope to their posterior means, surrounded by its 95% credible interval." width="90%" />
<p class="caption">(\#fig:nimble-fit-plot)Linear model fit with NIMBLE. The blue line is the estimated regression, obtained by setting the intercept and slope to their posterior means, surrounded by its 95% credible interval.</p>
</div>

Bayesian methods are often used for more complex models than linear regression (such as mixed models; see Chapter \@ref(glms)), for which there are no standard turnkey goodness-of-fit tests. In these situations, we commonly use what are called posterior predictive checks. The idea is to simulate new datasets from the posterior distribution of the model parameters, and then compare them to the observed data. The more the simulated data resemble the real data, the more it suggests that the model fits well. This comparison can be done visually or using a Bayesian p-value that quantifies the discrepancy between simulated and observed data.

In `brms`, you just do: 

``` r
pp_check(lm.brms)
```

<div class="figure" style="text-align: center">
<img src="05-regression_files/figure-html/ppcheck-brms-1.png" alt="Posterior predictive checks produced with brms. The black curve corresponds to the observed data; the blue curves to data simulated under the model. The x-axis shows the possible values of the simulated or observed response. The y-axis shows their estimated density." width="90%" />
<p class="caption">(\#fig:ppcheck-brms)Posterior predictive checks produced with brms. The black curve corresponds to the observed data; the blue curves to data simulated under the model. The x-axis shows the possible values of the simulated or observed response. The y-axis shows their estimated density.</p>
</div>

The `pp_check()` function generates posterior predictive check plots (Figure \@ref(fig:ppcheck-brms)). It compares observed data to data simulated from the fitted model. If the model fits the data well, then we should be able to use it to generate data that resemble the observed data. Therefore, if the simulated curves overlap the observations well, this indicates that the model captures the structure of the data correctly. Otherwise, this may suggest a model misspecification, for example an inappropriate link or distribution family (see Chapter \@ref(glms)).  

There is no dedicated function in `NIMBLE`, so we will need to simulate data under the model with the estimated parameters. We could do it by hand as with life expectancy, but the simplest approach is to include an additional line in the `NIMBLE` code: 

``` r
model <- nimbleCode({
  beta0 ~ dnorm(0, sd = 1.5) # normal prior on intercept
  beta1 ~ dnorm(0, sd = 1.5) # normal prior on coefficient
  sigma ~ dexp(1) # exponential prior on standard deviation
  for(i in 1:n) {
    y[i] ~ dnorm(beta0 + beta1 * x[i], sd = sigma) # model for observed data
    y_sim[i] ~ dnorm(beta0 + beta1 * x[i], sd = sigma) # model for simulated data
  }
})
```

This is the line `y_sim[i] ~ dnorm(beta0 + beta1 * x[i], sd = sigma)` that I added to simulate under the fitted model. The data and initial values do not change; we just need to add `y_sim` to the list of parameters we want to retrieve in the output: 

``` r
par <- c("beta0", "beta1", "sigma", "y_sim")
```

Then we rerun `NIMBLE`: 

``` r
lm.nimble <- nimbleMCMC(
  code = model,
  data = dat,
  inits = inits,
  monitors = par,
  niter = 2000,
  nburnin = 1000,
  nchains = 3
)
#> |-------------|-------------|-------------|-------------|
#> |-------------------------------------------------------|
#> |-------------|-------------|-------------|-------------|
#> |-------------------------------------------------------|
#> |-------------|-------------|-------------|-------------|
#> |-------------------------------------------------------|
```

We then merge the 3 chains, and select only the columns corresponding to `y_sim`: 

``` r
# merge
y_sim_mcmc <- rbind(lm.nimble$chain1, lm.nimble$chain2, lm.nimble$chain3)
# get columns corresponding to simulated y (y_sim[i])
y_sim_cols <- grep("^y_sim\\[", colnames(y_sim_mcmc))
# extract
y_sim_matrix <- y_sim_mcmc[, y_sim_cols]
```

We then take 10 draws, as `brms` does by default, and format the results: 

``` r
# set seed for reproducibility
set.seed(123)
# select at random 10 values
sim_indices <- sample(1:nrow(y_sim_matrix), 10)
# format simulated data
simulations_df <- data.frame(
  y_sim = as.vector(t(y_sim_matrix[sim_indices, ])), # sim values
  Replicate = rep(1:length(sim_indices), each = n), # id draw
  Observation = rep(1:n, times = length(sim_indices)) # id obs
)
```

Finally, we obtain the posterior predictive checks plot in Figure \@ref(fig:ppcheck-nimble): 

``` r
ggplot() +
  geom_density(aes(x = y_sim, group = Replicate), color = "lightblue", alpha = 0.2, data = simulations_df) +
  geom_density(aes(x = y), color = "black", alpha = 0.5, size = 1.2, data = data.frame(y = y)) +
  labs(x = "",
       y = "") +
  theme_minimal(base_size = 14)
```

<div class="figure" style="text-align: center">
<img src="05-regression_files/figure-html/ppcheck-nimble-1.png" alt="Posterior predictive checks produced with NIMBLE. The black curve corresponds to the observed data; the blue curves to data simulated under the model. The x-axis shows the possible values of the simulated or observed response. The y-axis shows their estimated density." width="90%" />
<p class="caption">(\#fig:ppcheck-nimble)Posterior predictive checks produced with NIMBLE. The black curve corresponds to the observed data; the blue curves to data simulated under the model. The x-axis shows the possible values of the simulated or observed response. The y-axis shows their estimated density.</p>
</div>

We can also compute a Bayesian p-value, which represents the proportion of datasets simulated under the model for which the chosen statistic (here the mean) is as large as or larger than the observed one. A value close to 0 or 1 can indicate a poor fit of the model for that particular statistic, whereas a value close to 0.5 suggests a good fit. This Bayesian p-value is obtained as follows: 

``` r
# Observed test stat
T_obs <- mean(y)

# Simulated test stat
T_sim <- apply(y_sim_matrix, 1, mean)

# Bayesian p-value: proportion of simulations where T_sim is more extreme than T_obs
bayes_pval <- mean(T_sim >= T_obs)
bayes_pval
#> [1] 0.512
```

With `brms`, we can also obtain this Bayesian p-value: 

``` r
# extract simulations
y_rep <- posterior_predict(lm.brms)

# compute test stat on sim data
T_sim <- rowMeans(y_rep)

# compute test stat on observed data
T_obs <- mean(lm.brms$data$y)

# compute Bayesian p-value
bayes_pval <- mean(T_sim >= T_obs)
bayes_pval
#> [1] 0.49075
```

## Model comparison

As we saw in Chapter \@ref(principles), Bayesian statistics makes it possible to compare several hypotheses with each other, and to assess how plausible a hypothesis is given the data we have collected.

Before comparing models, it is essential to ask what the goal of the analysis is: is it to better understand a phenomenon (an explanatory approach), or rather to make predictions (a predictive approach)?

One strategy is to build a single model that includes the variables deemed relevant, then fit it, examine it, test it, and improve it progressively. This approach aims less at identifying the best model than at exploring different variants to better understand the system under study.

To evaluate a model’s predictive ability, one can rely on data already used for fitting (internal prediction) or, more reliably, on new data (external prediction). The latter approach, however, requires splitting the data into a training set and a test set. If that is not possible, it is still possible to estimate predictive performance on the training data themselves using tools such as WAIC or LOO-CV.

WAIC (Watanabe–Akaike Information Criterion) and LOO-CV (Leave-One-Out cross-validation) allow models to be compared by estimating their ability to predict new data. They combine the fit to the observed data with a penalization for model complexity. A lower WAIC or LOO-CV value indicates a better model. WAIC is based on a theoretical approximation, whereas LOO-CV relies on cross-validation. LOO-CV is generally more accurate, especially for complex models or limited sample sizes, but it is also more computationally costly. In practice, when models are well specified and the sample is large, WAIC and LOO-CV often give very similar results for a given model.

Let us return to the linear regression example. We would like to test the hypothesis that the variable $x$ does explain an important part of the variation in $y$. This amounts to comparing models with and without this variable.



In `brms`, we fit these two models using weakly informative priors:

``` r
# Model with covariate
fit1 <- brm(y ~ x, data = data, family = gaussian(),
            prior = c(
              prior(normal(0, 1.5), class = Intercept),
              prior(normal(0, 1.5), class = b),
              prior(exponential(1), class = sigma)
            ))

# Model without covariate
fit0 <- brm(y ~ 1, data = data, family = gaussian(),
            prior = c(
              prior(normal(0, 1.5), class = Intercept),
              prior(exponential(1), class = sigma))
```

The function `waic()` can be used to extract the WAIC; the model with the smallest value is preferred. If the model with $x$ is indeed the correct one (which is what we expect since that is how the data were simulated), we should see that it is clearly better than the one without the covariate:

``` r
# Compute WAIC
waic1 <- waic(fit1)
waic0 <- waic(fit0)

# Compare
waic1$estimates['waic',]
#>  Estimate        SE 
#> 172.50456  13.13435
waic0$estimates['waic',]
#>  Estimate        SE 
#> 333.97491  17.23233
```

Phew, that is indeed the case. The function `loo()` can be used to compute the LOO-CV (an approximation, in fact):

``` r
# Leave-one-out cross-validation
loo1 <- loo(fit1)
loo0 <- loo(fit0)

# Compare
loo_compare(loo0, loo1)
#>      elpd_diff se_diff
#> fit1   0.0       0.0  
#> fit0 -80.7       9.1
```

In this `R` output, `elpd_diff` gives the difference in LOO-CV between each model and the one with the largest value. Thus, the best model is on the first line with an elpd_diff equal to zero; here, it is the model with the covariate. We therefore reach the same conclusion as with WAIC.

We can also obtain WAIC values with `NIMBLE`. To do so, we simply add `WAIC = TRUE` in the call to the function `nimbleMCMC`:

``` r
# Code of model with covariate
model_avec <- nimbleCode({
  # priors
  beta0 ~ dnorm(0, sd = 1.5)
  beta1 ~ dnorm(0, sd = 1.5)
  sigma ~ dexp(1)
  # likelihood
  for(i in 1:n) {
    y[i] ~ dnorm(beta0 + beta1 * x[i], sd = sigma)
  }
})

# Code of model without covariate
model_sans <- nimbleCode({
  # priors
  beta0 ~ dnorm(0, sd = 1.5) 
  sigma ~ dexp(1) 
  # likelihood
  for(i in 1:n) {
    y[i] ~ dnorm(beta0, sd = sigma) 
  }
})

# Data, initial values
dat <- list(x = x, y = y, n = n) # données
inits_avec <- list(list(beta0 = -0.5, beta1 = -0.5, sigma = 0.1), # inits chain 1
                   list(beta0 = 0, beta1 = 0, sigma = 1), # inits chain 2
                   list(beta0 = 0.5, beta1 = 0.5, sigma = 0.5)) # inits chain 3
inits_sans <- list(list(beta0 = -0.5, sigma = 0.1), # inits chain 1
                   list(beta0 = 0, sigma = 1), # inits chain 2
                   list(beta0 = 0.5, sigma = 0.5)) # inits chain 3

# Model with covariate
lm.avec <- nimbleMCMC(
  code = model_avec,
  data = dat,
  inits = inits_avec,
  niter = 2000,
  nburnin = 1000,
  nchains = 3,
  WAIC = TRUE)
#> |-------------|-------------|-------------|-------------|
#> |-------------------------------------------------------|
#> |-------------|-------------|-------------|-------------|
#> |-------------------------------------------------------|
#> |-------------|-------------|-------------|-------------|
#> |-------------------------------------------------------|

# Model without covariate
lm.sans <- nimbleMCMC(
  code = model_sans,
  data = dat,
  inits = inits_sans,
  niter = 2000,
  nburnin = 1000,
  nchains = 3,
  WAIC = TRUE)
#> |-------------|-------------|-------------|-------------|
#> |-------------------------------------------------------|
#> |-------------|-------------|-------------|-------------|
#> |-------------------------------------------------------|
#> |-------------|-------------|-------------|-------------|
#> |-------------------------------------------------------|
#>   [Warning] There are 1 individual pWAIC values that are greater than 0.4. This may indicate that the WAIC estimate is unstable (Vehtari et al., 2017), at least in cases without grouping of data nodes or multivariate data nodes.

# Compute WAIC
lm.avec$WAIC$WAIC
#> [1] 172.4424
lm.sans$WAIC$WAIC
#> [1] 333.3443
```

We reach the same conclusion as with `brms`. Note that `NIMBLE` does not directly provide a `loo()` function like `brms`, even though one could estimate LOO-CV “by hand”.

## In summary

+ Linear regression makes it possible to model the relationship between a continuous response variable and one or more explanatory variables, while accounting for residual variability.

+ Simulating data from a model is an excellent way to understand how it works and to test your code.

+ Weakly informative prior distributions (such as $N(0, 1.5)$ for the coefficients or $\text{Exp}(1)$ for $\sigma$) help constrain realistic values while still allowing the model the freedom to learn from the data.

+ Model validation and comparison can be performed using posterior predictive checks and criteria such as WAIC. These tools make it possible to evaluate model quality with respect to the data and to arbitrate between competing models.
