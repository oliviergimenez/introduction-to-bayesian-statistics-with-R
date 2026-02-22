# The Bayesian approach {#principes}

## Introduction

In this chapter, we lay the foundations by revisiting a few probability concepts that will be useful later on. I introduce the key ideas of Bayesian statistics through a simple example that helps fix ideas, and that we will often use throughout the book. We will also draw parallels between classical (or frequentist) statistics and Bayesian statistics.

## Bayes’ theorem

Let us not delay any further and get to the heart of the matter. Bayesian statistics relies on Bayes’ theorem (or Bayes’ formula), whose first formulation is attributed to the mathematician and Reverend Thomas Bayes. This theorem was published in 1763, two years after Bayes’ death, thanks to the efforts of his friend Richard Price. It was also discovered independently by Pierre-Simon Laplace.

Bayes’ theorem concerns conditional probabilities, which can sometimes be a bit tricky to understand. The conditional probability of an event A given an event B, denoted $\Pr(A \mid B)$, is the probability that A occurs, revised by taking into account the additional information that event B has occurred. For example, imagine that one of your friends rolls a (fair) die and asks you for the probability that the result is a six (A). Your answer is 1/6 because each face of the die has the same chance of appearing. Now, imagine that you are told that the number obtained is even (B) before you answer. Since there are only three even numbers, and only one of them is six, you can revise your answer: $\Pr(A \mid B) = 1/3$.

Do you see how the additional information (here, knowing that the number is even) changes the estimate? This is exactly the kind of reasoning that Bayes’ theorem formalizes and generalizes: it makes it possible to compute the probability of an event A given that another event B has occurred. More precisely, Bayes’ theorem gives you $\Pr(A \mid B)$ using the marginal probabilities $\Pr(A)$ and $\Pr(B)$ and the probability $\Pr(B \mid A)$:

$$\Pr(A \mid B) = \displaystyle{\frac{ \Pr(B \mid A) \; \Pr(A)}{\Pr(B)}}.$$

We talk about a marginal probability when we are interested in the probability of an event “on its own”, without any particular condition. For example, $\Pr(A)$ or $\Pr(B)$ are the overall chances of A or of B, without taking anything else into account. We say “marginal” because, if you made a table with all possible combinations (for instance, die outcomes classified as even/odd and as “six/not six”), then $\Pr(A)$ and $\Pr(B)$ would be obtained by adding the cells in a row or a column—i.e., what you read in the margin of the table.

Bayes’ theorem is often seen as a way to go from an effect B back to an unknown cause A, by knowing the probability of the effect B given the cause A. Think, for example, of a situation where a medical diagnosis is needed, with A an unknown disease and B symptoms; the physician knows the risks of having certain symptoms depending on several diseases, i.e. $\Pr(\text{symptoms}|\text{disease})$, and wishes to infer the probability of having a disease given the symptoms, i.e. $\Pr(\text{disease}|\text{symptoms})$. This way of “reversing” $\Pr(B \mid A)$ into $\Pr(A \mid B)$ is why Bayesian reasoning is sometimes called “inverse probability”.

Rather than using letters at the risk of getting confused, I find it easier to remember Bayes’ theorem written like this:

$$\Pr(\text{hypothesis} \mid \text{data}) = \frac{ \Pr(\text{data} \mid \text{hypothesis}) \; \Pr(\text{hypothesis})}{\Pr(\text{data})}.$$

The hypothesis can be a parameter such as the probability that a disease occurs, or regression coefficients linking this probability to risk factors (for example, place of residence, smoking). Bayes’ theorem tells us how to obtain the probability of a hypothesis from the available data.  
This is relevant because, think about it: this is exactly what the scientific method does. We want to know how plausible a hypothesis is given data that we have collected, and perhaps compare several hypotheses with one another. From this point of view, Bayesian reasoning aligns with scientific reasoning, which probably explains why the Bayesian framework feels so natural for doing and understanding statistics.

You might then ask why Bayesian statistics is not the norm. For a long time, implementing Bayes’ theorem was limited by computational difficulties, as we will see in the next chapter. Fortunately, increases in computing power and the development of new algorithms have led to a marked rise of the Bayesian approach over the past thirty years.

## What is Bayesian statistics? {#statbayes}

Typical statistical problems consist in estimating one (or several) parameters from available data. Let us denote this parameter (or these parameters) generically, say $\text{theta}$. To estimate $\text{theta}$, you are probably more familiar with the frequentist approach than with the Bayesian approach. The frequentist approach, in particular maximum likelihood estimation, assumes that parameters are fixed but unknown. Classical estimators are therefore generally point values; for instance, an estimator of the probability of obtaining a face of a die is the number of times that face was observed divided by the number of times the die was rolled. The Bayesian approach assumes that parameters are not fixed and follow an unknown distribution. A probability distribution is a mathematical expression that gives the probability that a random variable takes certain values. It can be discrete (for example, the Bernoulli distribution, the binomial distribution, or the Poisson distribution) or continuous (such as the normal or Gaussian distribution).

The Bayesian approach rests on the idea that you start with some knowledge about the system even before studying it yourself. Then, you collect data and update this prior knowledge based on the observations. This updating process relies on Bayes’ theorem. In simplified form, taking $A = \text{theta}$ and $B = \text{data}$, Bayes’ theorem makes it possible to estimate the parameter $\text{theta}$ from the data as follows:

$$\Pr(\text{theta} \mid \text{data}) = \frac{\Pr(\text{data} \mid \text{theta}) \times \Pr(\text{theta})}{\Pr(\text{data})}.$$

Let us take a moment to review each term in this formula.

On the left, we have $\Pr(\text{theta} \mid \text{data})$, the posterior distribution: the probability of $\text{theta}$ given the data. It represents what you know about $\text{theta}$ after seeing the data. This is the basis of inference and it is precisely what you are looking for: a distribution, possibly multivariate if you have several parameters.

On the right, we have $\Pr(\text{data} \mid \text{theta})$, the likelihood. The probability of the data given $\text{theta}$. This quantity is the same as in the classical or frequentist approach. Yes: Bayesian and frequentist approaches share the same component, the likelihood, which explains why their results are often close. The likelihood expresses the information contained in your data, given a model parameterized by $\text{theta}$. We will come back to it in Section \@ref(maxvrais).

Next, we have $\Pr(\text{theta})$, the prior distribution. This quantity represents what you know about $\text{theta}$ before seeing the data. This prior distribution should not depend on the data; in other words, one should not use the data to construct it. It can be vague or non-informative if you know nothing about $\text{theta}$. Often, you never really start from zero, and ideally you would like your prior to reflect existing knowledge. I will discuss priors in more detail in Chapter \@ref(prior).

Finally, there is the denominator $\Pr(\text{data})$, sometimes called the average likelihood, averaged with respect to the prior, because it is obtained by integrating the likelihood under the prior distribution:
${\Pr(\text{data}) = \int{\Pr(\text{data} \mid \text{theta}) \times \Pr(\text{theta}) \, d\text{theta}}}$.
This quantity normalizes the posterior distribution so that it integrates to 1. In other words, since $\int{\Pr(\text{theta} \mid \text{data}) \, d\text{theta}} = 1$ because the integral of a probability density equals 1, we have
$\displaystyle \int{\frac{\Pr(\text{data} \mid \text{theta}) \times \Pr(\text{theta})}{\Pr(\text{data})} \, d\text{theta} } = 1$.
And since $\Pr(\text{data})$ does not depend on $\text{theta}$, we have
$\Pr(\text{data}) = \int{\Pr(\text{data} \mid \text{theta}) \times \Pr(\text{theta}) \, d\text{theta}}$.
This is an integral whose dimension equals the number of parameters $\text{theta}$ to estimate: for two parameters, a double integral; for three parameters, a triple integral; and so on. However, beyond three dimensions, it becomes difficult, even impossible, to compute this integral. This is one of the reasons why the Bayesian approach was not used earlier, and why we need algorithms to estimate posterior distributions, as I explain in Chapter \@ref(mcmc). In the meantime, we will work through a relatively simple example in which the posterior distribution has an explicit form.

## A running example

Let us take a concrete example to fix ideas. I work on the coypu (*Myocastor coypus*) (Figure \@ref(fig:ragondinos)), a semi-aquatic rodent native to South America, introduced into Europe for fur farming. It is now considered an invasive alien species, because of the damage it causes in wetlands (bank erosion, destruction of vegetation) and its possible role in transmitting leptospirosis to humans, a potentially severe bacterial infection transmitted through water. Thanks to its high fecundity and good adaptation to temperate climates, the coypu has proliferated rapidly.

<div class="figure" style="text-align: center">
<img src="images/ragondin2.jpg" alt="Photograph of coypus (Myocastor coypus) taken in the Lez watershed near Montpellier, France. Credits: Yann Raulet." width="90%" />
<p class="caption">(\#fig:ragondinos)Photograph of coypus (Myocastor coypus) taken in the Lez watershed near Montpellier, France. Credits: Yann Raulet.</p>
</div>

One of the questions I am interested in is estimating the probability of surviving the winter, coypus being particularly sensitive to cold. To do this, we equip several individuals with a GPS tag at the beginning of winter, say here $n = 57$. At the end of winter, we observe that $y = 19$ coypus are still alive. The goal is to estimate the winter survival probability, which we denote $\text{theta}$. Here are the data:


``` r
y <- 19 # number of individuals that survived the winter
n <- 57 # number of individuals monitored at the start of winter
```

You are probably thinking that, with this information, we can already estimate a survival probability. Intuitively, we think of the proportion of individuals that survived, i.e. $19/57$. And you are not wrong. This is a reasonable estimate of $\text{theta}$, the winter survival probability. Let us now try to formalize this intuition, in order to better understand what it represents, and what it assumes.

As mentioned above, the likelihood is a central concept found in both frequentist and Bayesian approaches. So let us start by constructing this likelihood. To do that, we need to make a few assumptions.

First, we assume that individuals are independent, meaning that the survival of one coypu does not influence the survival of other coypus. This is a strong assumption, especially when we know that a female can reproduce two to three times per year and give birth to up to ten offspring that depend on her early in life. But, in modeling, it is often better to start simple.

Second, we assume that all individuals have the same survival probability. Again, this is a simplification: we know, for example, that juvenile mortality is higher than adult mortality.

Under these two assumptions, the number $y$ of animals still alive at the end of winter follows a binomial distribution, with $\text{theta}$ as the probability of success (survival) and $n$ as the number of trials (monitored individuals). We write $y \sim \text{Bin}(n, \text{theta})$. The binomial distribution is in fact the sum of several independent Bernoulli trials, as in the classic heads-or-tails example. At each trial—here, the release of a GPS-tagged coypu at the beginning of winter—we assume a probability $\text{theta}$ of success, i.e. surviving the winter, and failure, i.e. dying from cold. If all these trials are independent and have the same probability of success (our assumptions), then the number of successes, or the number of coypus alive at the end of winter, follows a binomial distribution (see also Chapter \@ref(glms)). I provide examples of Bernoulli and binomial draws in Figure \@ref(fig:bernoulli-binomiale).

<div class="figure" style="text-align: center">
<img src="01-principles_files/figure-html/bernoulli-binomiale-1.png" alt="Discrete probability distributions, Bernoulli and binomial, illustrated with 100 simulations (random draws generated by computer). On the top row, we show the observed frequency from a Bernoulli draw for different values of survival probability \(\theta\). On the bottom row, we show histograms for a binomial draw with 50 trials and different values of survival probability \(\theta\)." width="90%" />
<p class="caption">(\#fig:bernoulli-binomiale)Discrete probability distributions, Bernoulli and binomial, illustrated with 100 simulations (random draws generated by computer). On the top row, we show the observed frequency from a Bernoulli draw for different values of survival probability \(\theta\). On the bottom row, we show histograms for a binomial draw with 50 trials and different values of survival probability \(\theta\).</p>
</div>

As an aside, it is easy to get mixed up between all the terms used to describe a Bernoulli and a binomial distribution (and the normal distribution): you can remember that a probability is a number, a distribution is a law, and a density is the function that represents it.

## Maximum likelihood {#maxvrais}

In the classical (or frequentist) approach, we estimate the survival probability $\text{theta}$ using the maximum likelihood method. But what does that mean in practice? It means finding the value of $\text{theta}$ that makes the observed data most likely. In other words, since the data are what they are—they have been observed—we look for the value of $\text{theta}$ that maximizes the probability that this dataset was generated.

How do we justify this rather intuitive idea mathematically? Read carefully the end of the previous paragraph. The idea of looking for the value that gives the largest probability amounts to maximizing something. But what exactly? The probability of the data, given a certain model parameterized by $\text{theta}$—in other words, the likelihood, or $\Pr(\text{data}|\text{theta})$, which we saw in Section \@ref(statbayes). Classical estimation therefore relies on maximizing the likelihood—or rather the likelihood function, i.e. the likelihood considered as a function of $\text{theta}$.

In our case, we have a binomial experiment: we follow $n$ coypus over the winter, each having a probability $\text{theta}$ of surviving. We know the probability of each possible outcome (the probability mass function). For example, the probability that no coypu survives is $(1-\text{theta})^n$, because each of the $n$ individuals dies with probability $1-\text{theta}$. If we take, for example, a survival probability of 0.5, we have $(1-0.5)^{57} \approx 0$. We can compute this probability in R with the `dbinom()` function:


``` r
dbinom(x = 0, size = 57, prob = 1 - 0.5)
#> [1] 6.938894e-18
```

where the first argument `x = 0` corresponds to no coypu alive. Conversely, the probability that all survive is $\text{theta}^n$, which has the same value. You can check in `R` with `dbinom(x = 57, size = 57, prob = 0.5)`. If exactly one coypu survives, then one of the $n$ survives with probability $\text{theta}$, and the other $n-1$ die with probability $(1-\text{theta})^{n-1}$. Since any of the $n$ coypus can be the one that survives, we obtain a total probability of $n,\text{theta},(1-\text{theta})^{n-1}$. We can compute this probability with `dbinom(x = 1, size = 57, prob = 0.5)`. More generally, the probability that $y$ individuals survive is given by $\displaystyle \binom{n}{y}\text{theta}^y(1-\text{theta})^{n-y}$. If we consider this expression as a function of $\text{theta}$ (and not of $y$), we obtain the likelihood function $\displaystyle \mathcal{L}(\text{theta}) = \binom{n}{y} \text{theta}^y (1 - \text{theta})^{n - y}$. The term $\displaystyle \binom{n}{y}$ is called the binomial coefficient and is read “$y$ out of $n$”. It corresponds to the number of different ways to choose $y$ survivors among the $n$ coypus, without regard to their order.

We can plot this likelihood in `R` as in Figure \@ref(fig:survie-vraisemblance-mle):

<div class="figure" style="text-align: center">
<img src="01-principles_files/figure-html/survie-vraisemblance-mle-1.png" alt="Likelihood function for the winter survival probability of the coypu, computed from $y=19$ survivors out of $n=57$ individuals monitored by GPS. The maximum likelihood estimate is indicated by the red dashed line." width="90%" />
<p class="caption">(\#fig:survie-vraisemblance-mle)Likelihood function for the winter survival probability of the coypu, computed from $y=19$ survivors out of $n=57$ individuals monitored by GPS. The maximum likelihood estimate is indicated by the red dashed line.</p>
</div>

Our goal is to find the value of $\text{theta}$ that maximizes this function. In other words, we look for the survival value (on the x-axis in Figure \@ref(fig:survie-vraisemblance-mle)) that maximizes the likelihood (on the y-axis). This value corresponds to the maximum likelihood estimator, often denoted $\hat{\text{theta}}$. To do this, it is often more convenient to work with the logarithm of the likelihood (the log-likelihood), because sums are numerically more stable and easier to differentiate than products:

$$
\ell(\theta) = \log \mathcal{L}(\theta) = \log \binom{n}{y} + y \log \theta + (n - y) \log (1 - \theta).
$$
The first term, $\displaystyle \log \binom{n}{y}$, does not depend on $\text{theta}$, so we can ignore it in what follows. We then differentiate the log-likelihood with respect to $\text{theta}$:

$$
\displaystyle \frac{d\ell(\theta)}{d\theta} = \frac{y}{\theta} - \frac{n - y}{1 - \theta}.
$$

We look for the value of $\text{theta}$ that makes this derivative equal to zero:

$$
\frac{y}{\theta} - \frac{n - y}{1 - \theta} = 0.
$$

After a few simplifications, we obtain that the maximum likelihood estimator $\hat{\text{theta}}$ is:

$$
\hat{\theta} = \frac{y}{n}.
$$

This result matches our initial intuition: the maximum likelihood estimator is the proportion of individuals that survived, i.e. $19/57 \approx 0.333$. We can visualize this result in Figure \@ref(fig:survie-vraisemblance-mle), where the maximum likelihood estimate is indicated by the red dashed line.

In practice, models contain multiple parameters—dozens or even hundreds—and we cannot apply the same analytic method to maximize the likelihood and find the maximum likelihood estimators. Instead, we use iterative optimization algorithms that solve the problem for us, adjusting step by step an initial value until they find the one that maximizes the likelihood. For example, in R, we can obtain exactly the same result by using a logistic regression without covariates (see Chapter \@ref(glms)): 

``` r
mod <- glm(cbind(y, n - y) ~ 1, family = binomial)
theta_hat <- plogis(coef(mod))
theta_hat
#> (Intercept) 
#>   0.3333333
```

The direct calculation $\hat{\text{theta}}=y/n$ and the result of calling the glm function are consistent: they give the same value.

## And in the Bayesian framework?

In the Bayesian approach, we start by expressing our prior knowledge about the quantity we want to estimate—here, the winter survival probability `theta`. We know that `theta` is a continuous variable between 0 and 1. A natural prior distribution in this case is the beta distribution. The beta distribution is defined by two parameters, $a$ and $b$, which control its shape:

$$
q(\theta \mid a, b) = \frac{1}{\text{Beta}(a, b)}{\theta^{a - 1}} {(1-\theta)^{b - 1}}
$$

with:

$$
\text{Beta}(a, b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}, \quad \Gamma(n) = (n-1)!
$$

You can forget these equations if you are not comfortable with them. Let us instead visualize this distribution as in Figure \@ref(fig:beta-exemples):

<div class="figure" style="text-align: center">
<img src="01-principles_files/figure-html/beta-exemples-1.png" alt="Examples of beta distributions for different values of parameters $a$ and $b$. In each panel, the shaded areas illustrate the probability of observing a value within a given interval." width="90%" />
<p class="caption">(\#fig:beta-exemples)Examples of beta distributions for different values of parameters $a$ and $b$. In each panel, the shaded areas illustrate the probability of observing a value within a given interval.</p>
</div>

Each panel of the figure shows the shape of a beta distribution for a given pair of parameters $(a, b)$. Several characteristic behaviors can be observed.

- Beta(1,1) (top left) corresponds to the uniform distribution between 0 and 1: all values of theta between 0 and 1 are considered equally likely. The density is constant, which means that the probability of observing a value between 0.1 and 0.2 is the same as that of observing one between 0.8 and 0.9. This probability is the area of the rectangle bounded by the red curve and the vertical lines at 0.1 and 0.2 (or 0.8 and 0.9), i.e. the red shaded areas. This corresponds to a situation with no prior knowledge.
- Beta(2,1) and Beta(1,2) represent asymmetric knowledge: the former is biased toward values close to 1, the latter toward values close to 0. The probability of observing a value between 0.1 and 0.2 is smaller than that of observing one between 0.8 and 0.9, and vice versa.
- Beta(2,2) is symmetric but puts more weight on central values than a uniform distribution. The probability of observing a value between 0.1 and 0.2 is smaller than that of observing one between 0.5 and 0.6.
- Beta(10,10) represents knowledge that is highly concentrated around 0.5: it is a very informative prior. The probability of observing a value between 0.2 and 0.3 is much smaller than that of observing one between 0.5 and 0.6.
- Beta(0.8,0.8) illustrates a U-shaped (bathtub-shaped) distribution that favors extreme values (close to 0 or to 1). The probabilities of observing a value between 0 and 0.1 and between 0.9 and 1 are larger than that of observing one between 0.45 and 0.55.

These examples make it possible to visualize how parameters $a$ and $b$ influence the shape of the prior. How do we go from this prior to the posterior distribution?

We assume that $\theta \sim \text{Beta}(a, b)$ and we observed $y = 19$ survivors among $n = 57$ individuals. The likelihood is $\displaystyle \binom{n}{y}\theta^y(1 - \theta)^{n - y}$. For now, we will ignore the denominator $\Pr(y)$ in Bayes’ theorem; we will see in the next chapter why. Thus, the posterior is proportional to the product of the likelihood and the prior:
$\Pr(\theta \mid y) \propto \Pr(y \mid \theta) \times \Pr(\theta)$.
In our case, we multiply the likelihood and the prior term by term, and by rearranging the terms in $\theta$ and $1-\theta$, we obtain:

$$
\begin{aligned}
\Pr(\theta \mid y) &\propto \underbrace{\theta^y (1 - \theta)^{n - y}}_{\text{binomial likelihood}} \times \underbrace{\theta^{a - 1} (1 - \theta)^{b - 1}}_{\text{beta prior}} \\
&\propto \underbrace{\theta^{a + y - 1} (1 - \theta)^{b + n - y - 1}}_{\text{yet another beta distribution}}
\end{aligned}
$$

In other words, we again obtain a beta distribution, with updated parameters $a + y$ and $b + n - y$. We say that the binomial and beta distributions are conjugate: when we use a beta distribution as the prior for a probability parameter in a binomial model, the resulting posterior distribution is also a beta distribution. If we use a uniform prior between 0 and 1 (i.e. Beta(1,1)), we obtain that the posterior distribution of winter survival is
$\text{Beta}(1+19, 1+57-19) = \text{Beta}(20, 39)$.
Moreover, the posterior distribution is known, which greatly facilitates computations and interpretation. For example, we know that the mean of $\text{Beta}(a, b)$ is $\displaystyle \frac{a}{a+b}$, i.e. $\frac{20}{59} \approx 0.339$. We can compare this value to the maximum likelihood estimator $19/57 \approx 0.333$. We can also visualize the posterior distribution as in Figure @ref(fig:posterior-survie), since we know the equation of the beta density:

<div class="figure" style="text-align: center">
<img src="01-principles_files/figure-html/posterior-survie-1.png" alt="Distribution a priori uniforme (rouge) et distribution a posteriori (noire) de la probabilité de survie hivernale du ragondin. Le pointillé bleu correspond à l'estimateur du maximum de vraisemblance." width="90%" />
<p class="caption">(\#fig:posterior-survie)Distribution a priori uniforme (rouge) et distribution a posteriori (noire) de la probabilité de survie hivernale du ragondin. Le pointillé bleu correspond à l'estimateur du maximum de vraisemblance.</p>
</div>

More generally, when we have enough data, Bayesian and frequentist estimators tend to be very close. Intuitively, the data end up “dominating” the prior information. Roughly speaking, the mode of the posterior distribution (the value at which the density is maximal) corresponds exactly to the maximum likelihood estimator.

This illustrates the link between the two approaches and the central role of the likelihood in statistics: it is the fundamental common component of Bayesian and frequentist approaches.

## In summary

- Bayes’ theorem is a tool for updating knowledge.
- Bayesian statistics relies on the likelihood and a prior distribution for the model parameters.
- Frequentist statistics provides a point estimator, whereas Bayesian statistics estimates a distribution for each parameter.
- Often, classical and Bayesian approaches yield similar estimates.
- In some cases, the posterior distribution is explicit (for example, in the case of beta/binomial conjugacy).
- In most cases, we will need to use simulations to obtain the posterior distribution, as we will see in Chapter \@ref(mcmc).

