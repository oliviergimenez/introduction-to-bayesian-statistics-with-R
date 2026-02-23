# Prior distributions {#prior}

## Introduction

In this chapter, we will explore a fundamental aspect of Bayesian statistics: the role of prior distributions, or priors. We will see how these priors interact with the data via Bayes’ theorem to produce the posterior distribution, and how this influence varies depending on how much information the data provide. We will also learn how to incorporate relevant external information from expert knowledge or previous studies, and how to critically assess our prior choices using simulations.

## The role of the prior {#roleprior}

In Bayesian statistics, the prior plays an essential role: it expresses our knowledge, our uncertainties, or, conversely, our lack of information about the parameters of a model. Choosing priors well is therefore a key step in any Bayesian analysis. Why use a prior?

- To incorporate existing knowledge: we often have information from previous studies, meta-analyses, or expert opinion. The prior makes it possible to formalize and include this prior knowledge, rather than ignoring it and acting as if we were starting from nothing. We will see an example in Section \@ref(informativeprior). 

- To deal with a lack of data: when data are scarce or not very informative, frequentist methods can fail to estimate certain parameters correctly (boundary estimates for a probability, or a random-effect variance estimated as zero). In these situations, a well-chosen prior can help stabilize inference by providing complementary information.

- To constrain complex models: in mixed models, or in the presence of parameters that are difficult to estimate, priors make it possible to bound the solution space to plausible values and avoid aberrant estimates. For example, in a mixed model (see Chapter \@ref(glms)) where we estimate the variance between groups or levels of an effect, the absence of a prior can lead to unrealistic values or numerical instabilities. A weakly informative prior can help in this situation.

- To prevent overfitting: in models with many explanatory variables, priors play a regularization role by penalizing unimportant effects. For example, in a regression that includes many covariates, a prior of the form $N(0,1.5^2)$ prevents the model from assigning overly strong effects to weakly informative variables, thereby reducing the risk of overfitting.

The choice of a prior depends directly on the context and the scientific question.

- A non-informative prior aims to express a lack of knowledge: it is often used when one does not want to introduce strong assumptions. In practice, this translates into wide or uniform distributions. But beware: even a seemingly vague prior can be informative once transformed to the model scale, as we will see in Section \@ref(surprise). 

- An informative prior reflects credible knowledge external to the dataset being analyzed: it may come from a literature synthesis, past experience, or expert opinion. Its advantage is to reduce uncertainty on parameters, especially with little data. We will see an example in Section \@ref(informativeprior). 

- A weakly informative prior is somewhat a compromise between non-informative and informative priors. The idea is to rule out values that are clearly aberrant or incompatible with what we know about the phenomenon being studied, while still leaving enough freedom for the model to learn from the data. This type of prior is used notably in `brms`. We will see an example in Chapter \@ref(glms). 

In practice, a cautious strategy is to start with a weakly informative prior, such as a centered normal distribution with moderate variance, then to test more informative (or more vague) alternatives to examine the impact on posterior results. This is the idea of sensitivity analysis developed in Section \@ref(sensibilite). 

## Sensitivity to the prior {#sensibilite}

Let us return to our running example on coypu survival. Let us examine how different choices of priors influence the posterior distribution of this survival probability. In Figure \@ref(fig:priors-comparaison), we have three increasingly informative priors (in columns), and two sample sizes (in rows).

<div class="figure" style="text-align: center">
<img src="04-priors_files/figure-html/priors-comparaison-1.png" alt="Combined effect of the prior and sample size on the posterior distribution with a binomial likelihood. Columns: three beta priors Beta(1,1), Beta(5,5) and Beta(20,1). Rows: small (n = 6, y = 2) and large (n = 57, y = 19) sample (factor 10). The red line represents the prior, the black line the posterior distribution." width="100%" />
<p class="caption">(\#fig:priors-comparaison)Combined effect of the prior and sample size on the posterior distribution with a binomial likelihood. Columns: three beta priors Beta(1,1), Beta(5,5) and Beta(20,1). Rows: small (n = 6, y = 2) and large (n = 57, y = 19) sample (factor 10). The red line represents the prior, the black line the posterior distribution.</p>
</div>

With little data (top row), the effect of the prior is visible: the posterior distribution of survival remains close to the prior, especially with the $\text{Beta}(20,1)$ which pulls the estimate toward high values. With more data (bottom row), the posterior distribution is dominated by the likelihood: it concentrates around the observed proportion, except for the prior $\text{Beta}(20,1)$ for which the posterior distribution is centered on 0.5. We thus observe a fundamental principle of Bayesian inference: the more numerous and informative the data are, the less the prior influences the results.

We can formalize the observations made in Figure \@ref(fig:priors-comparaison). Recall that when the likelihood is $\text{Bin}(n,\theta)$ with $y$ successes, and the prior is a $\text{Beta}(a,b)$ distribution, the posterior distribution is also beta (conjugacy), and more precisely $\text{Beta}(a+y,\;b+n-y)$. Now, the mean of a $\text{Beta}(a,b)$ is $\displaystyle \frac{a}{a+b}$, and therefore the mean of the posterior distribution $\text{Beta}(a+y,\;b+n-y)$ is $\displaystyle \frac{a+y}{a+b+n}$, which can be rewritten as a weighted average between the mean of the prior distribution $\mu_{prior} = \displaystyle \frac{a}{a+b}$ and the observed proportion $y/n$, which is none other than the maximum likelihood estimator $\hat{\theta}$, with weight $w = \displaystyle \frac{n}{a+b+n}$. Note: this is a weight in the statistical sense of the term, a weighting factor, not in the sense of “kilograms of coypu”. In other words, the mean of the posterior distribution is $(1-w)\mu_{prior} + w \hat{\theta}$. Thus, when the sample size $n$ is large, $w$ tends to 1, and the posterior mean approaches the maximum likelihood estimator. Conversely, for a small sample or a very informative prior (the sum $a+b$ is large; see Figure \@ref(fig:beta-exemples)), $w$ is small, and the prior pulls the estimate. In short, when data are limited, we rely more on the prior; when they are rich, we let the likelihood speak.

In conclusion, it is always a good idea to carry out this kind of sensitivity analysis. By comparing results obtained with different priors (non-informative, weakly informative, informative), we can ensure that conclusions do not depend excessively on prior choices. If they do, do not panic: it simply means we have little information about the parameter in question, and we must be extra cautious and think carefully about the prior used. We will return to this later.

## How to incorporate prior information? {#informativeprior}

### Meta-analysis

Let us go back to our running example on estimating a survival probability, but making it slightly more complex to account for a common issue when studying animal populations: imperfect detection of individuals. Indeed, depending on behavior or field conditions, an animal may very well be alive and present, but not detected at the time of sampling. To correct this bias, capture–recapture protocols are often used, which rely on individual identification of animals, via a ring, a coat pattern, a genetic profile, etc.

An individual can thus be detected (1) or not (0), and we code for example 101 which means: seen the first year, missed the second, then seen again the third. In the simplest model, we assume a constant survival probability \(\theta\) and a constant detection probability \(p\). The likelihood for history 101 is therefore: $\Pr(101)=\theta\,(1-p)\,\theta\,p$. To obtain the full likelihood, we perform this calculation for each individual and assume that all share the same \(\theta\) and \(p\), and that they are independent.

To take a break from coypus, let us look at the White-throated Dipper (*Cinclus cinclus*), a bird studied for more than 40 years by Gilbert Marzolin, a mathematics teacher passionate about ornithology with whom I had the chance to work. We have capture–recapture data here over 7 years (1981–1987) for more than 200 birds.

We will start with a non-informative prior on survival probability, say a $\text{Beta}(1,1)$. This will be our model A. As an alternative prior, we can draw on accumulated knowledge for similar species. In passerines, for instance, there is a relationship between body mass and survival probability: on average, heavier birds live longer. This allometric relationship was quantified by @mccarthy2007 via a linear regression (see Chapter \@ref(lms)), based on survival and mass data for 27 European passerine species. Using this regression for passerines in the specific case of the dipper, and knowing that the dipper weighs on average 59.8 grams, we can predict its annual survival probability. The model thus provides an estimate of 0.57 with a standard error of 0.075. These values allow us to define an informative prior, in the form of a normal distribution centered at 0.57 with variance $0.075^2$. This will be our model B.

We thus obtain the following results for the dipper:

| Model | Prior for \(\theta\) | Posterior mean survival | 95% credible interval |
|-----|---------|--------------|--------------|
| A     | Beta(1,1)  | 0.56                      | [0.51 ; 0.61] |
| B     | N(0.57, 0.075²)| 0.56                      | [0.52 ; 0.61] |

With a rich dataset (7 years), the information contained in the likelihood dominates; the informative prior adds almost no information, and the two models produce very similar results.

Now imagine that we have limited data. What happens if we only have the first three years, for example? We redo the analysis, and the results are now:

| Model | Prior for \(\theta\) | Posterior mean survival | 95% credible interval |
|-----|---------|--------------|--------------|
| A  | Beta(1,1) | 0.70 | [0.47 ; 0.95] |
| B  | N(0.57, 0.075²) | 0.60 | [0.48 ; 0.72] |

This time, the informative prior makes a real difference. The width of the interval is reduced by nearly 50%, while bringing the mean estimate back toward a more realistic value for a passerine. We also note that the posterior estimate of model B with 3 years of data is close to that obtained with 7 years (Figure \@ref(fig:comparaison-prior-survie)).

<div class="figure" style="text-align: center">
<img src="04-priors_files/figure-html/comparaison-prior-survie-1.png" alt="Comparison of posterior estimates of dipper survival according to the type of prior and study duration. Each point represents the posterior mean, with its 95% credible interval. The grey line indicates the survival value from the meta-analysis for passerines (0.57)." width="100%" />
<p class="caption">(\#fig:comparaison-prior-survie)Comparison of posterior estimates of dipper survival according to the type of prior and study duration. Each point represents the posterior mean, with its 95% credible interval. The grey line indicates the survival value from the meta-analysis for passerines (0.57).</p>
</div>

This example shows that information from the literature (here an allometric mass–survival relationship obtained via a meta-analysis) can be used to build a relevant informative prior, capable of substantially improving the precision of estimates, especially when data are limited. This approach offers a low-cost alternative to lengthening field protocols, provided of course that the (relatively simple) question remains the estimation of a single survival.





### Moment-matching method

In the dipper example, we used a normal distribution as an informative prior for a parameter that happens to be a probability. However, the normal distribution can take negative values or values greater than 1, which is not desirable for a probability. In the example, the informative prior $N(0.57, 0.075^2)$ is on average between 0 and 1 with a small variance, so there is little chance that this goes wrong. You can see this by simulating values in `R` with the command `summary(rnorm(n = 100, mean = 0.57, sd = 0.075))`. Still, it is not very satisfying.

The good news is that we can construct a more appropriate informative prior for a probability using the so‑called “moment-matching” method. The moment-matching method consists in choosing the parameters of a prior distribution by matching the moments (often the mean and the variance) that represent the prior information we have (before seeing the data).

When the prior information is available in the form of a mean \(\mu\) and a standard deviation \(\sigma\), we can transform these moments into parameters \(a,b\) of a beta distribution. As a reminder, the mean and the variance of a beta distribution with parameters $a$ and $b$ are \(\mu=\dfrac{a}{a+b}\) and \(\sigma^2=\dfrac{ab}{(a+b)^2(a+b+1)}\). By inverting these relationships, we obtain: \(a=\displaystyle \Bigl(\frac{1-\mu}{\sigma^2}-\frac{1}{\mu}\Bigr)\mu^2\) and \(b=\displaystyle a\Bigl(\frac{1}{\mu}-1\Bigr)\). In our example, we have \(\mu=0.57\) and \(\sigma=0.075\), from which we can deduce \(a = 24.3\) and \(b = 18.3\) with a few lines of code:

``` r
# desired mean and standard deviation for the beta distribution
mu <- 0.57 # mean probability
sigma <- 0.075 # standard deviation on that probability
# inverse formulas to obtain the parameters a and b of a beta distribution
a <- ((1 - mu) / (sigma^2) - 1 / mu) * mu^2
b <- a * (1 / mu - 1)
# display a and b rounded
c(a = round(a, 1), b = round(b, 1))
#>    a    b 
#> 24.3 18.3
```

We can check that this beta distribution indeed has the mean and standard deviation given by the meta-analysis:

``` r
# generate 10,000 values from a Beta distribution with parameters a = 24.3 and b = 18.3
ech_prior <- rbeta(n = 10000, shape1 = 24.3, shape2 = 18.3)
# empirical mean of the draws (should be close to 0.57)
mean(ech_prior)
#> [1] 0.5685004
# empirical standard deviation of the draws (should be close to 0.075)
sd(ech_prior)
#> [1] 0.07496597
```

We can therefore adopt a prior \(\text{Beta}(a=24.3,\,b=18.3)\) to incorporate the mean information and its variability obtained from the allometric survival–mass relationship.

The moment-matching method does not apply only to probabilities. It can also be used to construct a prior for a real-valued parameter, for example the effect of coypu body mass on survival (see Chapter \@ref(lms)). Suppose an expert says: “I am 80% sure that parameter $\theta$ lies between –0.15 and 0.25.” This sentence defines an 80% credible interval: $\Pr(\theta \in [-0.15,0.25]) = 0.80$. We seek a normal prior $\theta \sim N(\mu,\sigma^2)$ that reflects exactly this information.

We can start with the mean \(\mu\). The interval is symmetric, so we can directly deduce that the mean \(\mu\) of the prior is the midpoint of the interval: \(\displaystyle{\mu = \frac{-0.15+0.25}{2}}=0.05\).

Now let us move to the standard deviation \(\sigma\). The expert states that 80% of the values of \(\theta\) are between –0.15 and 0.25. For a normal distribution, this proportion can be written as \(\Pr(\mu - z\,\sigma \leq \theta \leq \mu + z \, \sigma) = 0.80\). This means that 80% of the mass of the distribution is contained in an interval centered on \(\mu\) and of width \(2z\sigma\). For a level of 80%, the value of \(z\) is about 1.2816 (obtained via `qnorm(0.90)`, where 0.90 is the upper quantile \(1−\alpha/2 = 1-20/2\) with \((1−\alpha)\% = 80\%\) and thus \(\alpha = 0.20\)). Finally, we obtain \(\sigma = \displaystyle \frac{0.25-(-0.15)}{2 \times 1.2816} \approx 0.156\). Here is the calculation in `R`:

``` r
# lower and upper bounds given by the expert
a <- -0.15
b <-  0.25

# stated confidence level
level <- 0.80
alpha <- 1 - level

# z value corresponding to an 80% credible interval
z <- qnorm(1 - alpha / 2)  # ≈ 1.2816

# mean = center of the interval
mu <- (a + b) / 2

# standard deviation deduced from the interval width
sigma <- (b - a) / (2 * z)

mu
#> [1] 0.05
sigma
#> [1] 0.1560608
```

We conclude that the desired informative prior is \(N(\mu=0.05,\sigma=0.156)\). We can check that everything went well:

``` r
mu    <- 0.05
sigma <- 0.1560608
pnorm(c(-0.15, 0.25), mean = mu, sd = sigma)
#> [1] 0.09999996 0.90000004
#> 0.10 0.90    # OK: 10% on the left, 90% on the right → 80% in the center
```

Visually, Figure \@ref(fig:prior-normal-viz) shows the density of a normal distribution with mean \(\mu=0.05\) and standard deviation \(\sigma=0.156\). The light-blue interval corresponds to the central 80% credible interval, that is, the interval [−0.15; 0.25] which contains 80% of the probability mass. The grey dotted lines indicate the bounds of this interval, while the black dashed line marks the position of the mean. We see that, thanks to the symmetry of the normal distribution, the interval is centered around the mean, and that 10% of the mass lies on each side outside this interval.

<div class="figure" style="text-align: center">
<img src="04-priors_files/figure-html/prior-normal-viz-1.png" alt="Normal distribution with mean 0.05 and standard deviation 0.156. The shaded interval corresponds to the 80% credible interval, between –0.15 and 0.25." width="100%" />
<p class="caption">(\#fig:prior-normal-viz)Normal distribution with mean 0.05 and standard deviation 0.156. The shaded interval corresponds to the 80% credible interval, between –0.15 and 0.25.</p>
</div>

## Beware of so-called non-informative priors {#surprise}

In Bayesian statistics, we often use non-informative priors. But be careful: appearances can be misleading, especially when working with parameters defined on transformed scales, such as the logit or the log in generalized linear models (Chapter \@ref(glms)). Let us take a common example where we model a probability $\theta$ on the logit scale via a parameter $\beta$ such that $\text{logit}(\theta) = \beta$.

In practice, we can use simulations to check that priors do not bring unpleasant surprises after transformation; this is what we call prior predictive checks. This happens even before fitting a model, and to do so we will:

1. simulate values from the prior of $\beta$ on the logit scale;
2. apply the inverse logit transformation to obtain $\theta$;
3. inspect the induced prior distribution on $\theta$ and judge whether it seems realistic.

A first choice is to take as a prior a normal distribution with a large variance, for example $\beta \sim N(0, 10^2)$. Steps 1 and 2 are obtained via:

``` r
logit_prior <- rnorm(n = 1000, mean = 0, sd = 10) # simulation
prior <- plogis(logit_prior) # transformation
```

The problem is that after transformation with the inverse logit function, most simulated values—and thus the probability $\theta$—are close to 0 or 1 as we see in Figure \@ref(fig:prior-combined-ggplot) (left panel), which implicitly favors extreme values. We go from a non-informative prior on the logit scale to a very informative prior (without meaning to) on the natural scale of the probability.

Another choice is to take $\beta \sim N(0, 1.5^2)$. The first two steps of the simulation can be summarized as:

``` r
logit_prior2 <- rnorm(n = 1000, mean = 0, sd = 1.5)
prior2 <- plogis(logit_prior2)
```

Here the induced distribution on $\theta$ is uniform, covering mainly the range of values between 0.05 and 0.95 as we can see in Figure \@ref(fig:prior-combined-ggplot) (right panel), which better reflects a lack of information about $\theta$. This second choice is the right one; we speak of weakly informative priors.

<div class="figure" style="text-align: center">
<img src="04-priors_files/figure-html/prior-combined-ggplot-1.png" alt="Comparison of two priors obtained for the probability \( \theta = \text{logit}^{-1}(\beta) \) after transformation by the inverse logit function of \( \beta \sim N(0, 10^2) \) and \( \beta \sim N(0, 1.5^2) \). The x-axis represents the different possible values of the probability \( \theta \) obtained after transformation by the inverse logit. The y-axis indicates the frequency of simulated draws for each value." width="100%" />
<p class="caption">(\#fig:prior-combined-ggplot)Comparison of two priors obtained for the probability \( \theta = \text{logit}^{-1}(\beta) \) after transformation by the inverse logit function of \( \beta \sim N(0, 10^2) \) and \( \beta \sim N(0, 1.5^2) \). The x-axis represents the different possible values of the probability \( \theta \) obtained after transformation by the inverse logit. The y-axis indicates the frequency of simulated draws for each value.</p>
</div>

There are also invariant priors, that is, priors whose shape accounts for the scale of the parameter. Jeffreys' prior is an example: it maximizes the information brought by the data, while remaining invariant under reparameterization. For example, for a probability $\theta$, Jeffreys' prior is $\text{Beta}(0.5, 0.5)$. This prior is less flat than a uniform $\text{Beta}(1, 1)$. It is often used when one wants an objective approach, without introducing subjective information. In practice, however, Jeffreys' prior is difficult to compute, and we will prefer the simulation-based approach to ensure that transformed parameters have reasonable priors.

## Summary

+ The richer the data are, the less the prior influences the posterior estimate.

+ Do not hesitate to take the time to visualize your priors on the natural scale of the parameters using simulations.

+ Moment-matching methods offer a practical way to transform and encode knowledge into the parameters of distributions that can serve as priors (beta or normal, for example).

+ When should you use which type of prior?

| Type of prior            | Recommended use                                                                 | Advantages                                                              | Precautions                                                              |
|--------------------------|----------------------------------------------------------------------------------|-------------------------------------------------------------------------|--------------------------------------------------------------------------|
| Informative              | When one has solid information (expertise, meta-analysis, etc.)                 | Incorporates available knowledge, useful with little data               | Risk of bias if poorly calibrated                                        |
| Weakly informative       | By default if one wants to guide inference without constraining it              | Protects against implausible values, improves numerical stability       | Must be adapted to the scale of the parameter                            |
| Non-informative          | Exploratory cases, or to let the data speak                                     | Does not a priori favor any particular value                            | Can be misleading on transformed scales (logit, log)                     |
| Reference / Jeffreys     | When one seeks an invariant approach (a $\text{Beta}(0.5,0.5)$ in the running example) | Invariant under changes of parameterization                             | Sometimes difficult to compute or to interpret                           |
