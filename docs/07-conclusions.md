# Conclusions {-}

## What we covered {-}

I hope that with this book I have (at least a little) demystified Bayesian statistics and MCMC methods. I also hope to have given you the tools to understand the difference between frequentist and Bayesian approaches, to better read the “Methods” section of papers using Bayesian statistics, and to gain a certain level of autonomy in conducting Bayesian analyses.

Throughout the book, we covered several essential steps. We began by exploring the motivations for using the Bayesian approach. We then introduced Bayes’ theorem and discussed its interpretation. We discovered Markov chain Monte Carlo (MCMC) methods, and then worked with two powerful tools, `NIMBLE` and `brms`, to fit complex models. Particular attention was given to the role of prior distributions, whether non-informative or informative, as well as to the use of these approaches in case studies involving GLM and GLMM.

## Bayesian statistics, in a nutshell {-}

The Bayesian approach offers many advantages. It allows uncertainty to be quantified coherently using probability, it enables the explicit integration of prior knowledge, and it makes it possible to fit complex models via MCMC. In addition, Bayesian credible intervals are more intuitive than frequentist confidence intervals.

Some caution is nevertheless required. Checking the convergence of MCMC chains is a crucial step, but sometimes a laborious one. The choice of prior distributions requires careful consideration. Model fit must always be evaluated. Finally, the computational cost is not negligible, especially for the most complex models and/or large datasets.

## A few tips {-}

Before finishing, I would like to leave you with a few tips inspired by my own experience. These tips are not necessarily specific to Bayesian statistics, and they are worth what they are worth.

First, take the time to clearly formulate your question. This may seem obvious, but this step is essential to stay on track and make the right choices, for example deciding to use only a subset of the data to answer a specific question.

Next, think about your model first, and formalize it either with equations, by drawing it, or in words. What is the nature of your data, and therefore, if you are in a regression framework, which family of distributions should you use as we saw in Chapters \@ref(lms) (normal) and \@ref(glms) (Bernoulli/binomial and Poisson)? Do not rush to the keyboard. Make sure you understand it, for example by explaining it to colleagues.

On that note, remember to run simulations. Simulating data from your model often helps you understand it better, as in Chapters \@ref(lms) and \@ref(glms). It is an excellent way to test your assumptions and diagnose potential issues.

Choose the `R` environment you are comfortable with; I illustrated `brms` and `NIMBLE` (Chapter \@ref(mcmc)), but other solutions exist.

When fitting the model, start simple. A model with all parameters constant is a good baseline. This ensures that the data are read and formatted correctly, that there are no outliers (an extra zero, a misplaced comma), and that the priors do not generate unexpected behavior (see Chapter \@ref(prior)). This approach is particularly important in Bayesian statistics to ensure good performance and convergence of the MCMC algorithm (Chapter \@ref(mcmc)), while also giving you an idea of the time required to run the analysis. Once everything looks good, gradually add complexity, random effects for example (Chapter \@ref(glms)), until you reach the model structure that seems most appropriate to answer your question. This likely implies several iterations of fitting, comparing, and validating your models (Chapters \@ref(lms) and \@ref(glms)).

For further practical guidance, I recommend reading the papers “Ten quick tips to get you started with Bayesian statistics” [@gimenez2025] and “Bayesian workflow” [@gelman2020bayesian].

## To conclude {-}

Adopt a pragmatic approach. The choice of statistical approach (frequentist or Bayesian) depends on your objectives, whether they concern speed, model complexity, or the type of uncertainty you want to quantify. Discuss your options with more experienced colleagues if needed. Bayesian statistics is not a dogma: it is a powerful tool among others in your toolbox.

Thank you for your attention. Feel free to write to me if you have questions or if you would like to see a particular aspect developed in a new edition of this book. And enjoy exploring Bayesian statistics!
