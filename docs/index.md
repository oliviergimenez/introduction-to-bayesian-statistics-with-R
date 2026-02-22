---
title: "Introduction to Bayesian Statistics with R"
subtitle: "Using NIMBLE and brms"
author: "Olivier Gimenez"
date: "2026-02-22"
knit: "bookdown::render_book"
site: bookdown::bookdown_site
output:
  bookdown::bs4_book:
    downloads:
      - name: "Télécharger le PDF"
        url: statistique-bayesienne.pdf
  bookdown::pdf_book:
    latex_engine: xelatex
    toc: true
    number_sections: true
    keep_tex: false
documentclass: book
bibliography: [book.bib]
biblio-style: apalike
link-citations: true
links-as-notes: true
colorlinks: true
url: https://oliviergimenez.github.io/introduction-to-bayesian-statistics-with-R/
description: "Introduction to Bayesian statistics with R"
---



# Introduction {.unnumbered}

Bayesian statistics can be found almost everywhere in science. For example, in epidemiology to predict the spread of viruses, in ecology to understand the extinction of plant and animal species, or in computer science to filter unwanted emails. The widespread adoption of Bayesian methods over the past decades has largely been driven by advances in computing power. But it is also due to the nature of the approach itself, which closely matches how we learn, reason, and accumulate knowledge.

In this book, I offer an introduction to Bayesian statistics. You are currently reading the electronic version of the book, which is the English version of a book published by Quae in March 2026.

<img src="images/cover.png" width="100%" style="display: block; margin: auto;" />

My goals in writing this book were twofold:\
1) to synthesize the key methodological concepts that are essential to understand, and\
2) to provide practical tools so that you can apply Bayesian statistics yourself.

Because we learn best by doing, we will use software to practice statistics. That software is `R`, a free and open-source environment for statistical computing and data science.

For Bayesian analysis specifically, I present two practical tools:

-   `brms`, which provides a simple and familiar syntax similar to classical regression modeling in R;\
-   `NIMBLE`, which requires more programming but offers great flexibility.

Rather than adopting a formal academic style, I chose to write this book as if we were in the same room - or on a video call - and I were explaining Bayesian statistics to you directly. As a result, I will occasionally (and sometimes often) use informal language and mathematical shortcuts to make the ideas easier to grasp. I hope you will not mind.

## Why learn Bayesian statistics? {.unnumbered}

Bayesian statistics provides a framework for analyzing data and making decisions under uncertainty, much like predicting the weather or rolling a die: we cannot know exactly what will happen, but we can estimate the probability of different outcomes.

Why adopt this approach? Several reasons may motivate its use:

-   **A natural interpretation of probability**: in Bayesian statistics, probability represents a degree of belief in a hypothesis or parameter, aligning well with how we intuitively reason under uncertainty;\
-   **Great flexibility**: the Bayesian framework handles incomplete, heterogeneous, or scarce data, as well as complex models (hierarchical, nonlinear, dynamic, etc.);\
-   **Integration of prior knowledge**: previous studies or expert knowledge can be incorporated transparently and formally;\
-   **Explicit uncertainty quantification**: Bayesian inference provides not only parameter estimates but also direct measures of uncertainty.

## What you will learn in this book {.unnumbered}

My goal is to guide you through the learning process of Bayesian statistics. I have gathered the material that I consider essential for understanding and applying the approach. By the end, you should feel comfortable using Bayesian methods with your own data.

The objectives are to:

-   demystify Bayesian statistics and Markov chain Monte Carlo (MCMC) methods;\
-   understand the differences between Bayesian and frequentist approaches;\
-   read and interpret the “methods” sections of scientific articles using Bayesian analysis;\
-   implement Bayesian analyses in `R`.

Chapter \@ref(principles) introduces the foundations, revisiting key probability concepts and presenting core ideas through a simple example.

Chapter \@ref(mcmc) takes you behind the scenes of Bayesian inference, explaining MCMC methods and guiding you through coding your own Bayesian analysis.

Chapter \@ref(software) introduces two powerful tools for Bayesian modeling: `NIMBLE` and `brms`.

Chapter \@ref(prior) focuses on prior distributions—how to choose them, incorporate existing knowledge, and avoid common pitfalls.

Chapter \@ref(lms) presents Bayesian linear regression, including model comparison and validation, with examples using both `NIMBLE` and `brms`.

Chapter \@ref(glms) extends the discussion to generalized linear models, with and without random effects, illustrated through simulated data.

Finally, the last chapter summarizes the key take-home messages and offers practical advice for applying Bayesian statistics.

## How to read this book {.unnumbered}

There is no single “best” way to read this book. Personally, I always find it difficult to absorb all the information in a technical book in one pass. You may read it sequentially or dip into specific sections as needed.

In each chapter, the `R` code is provided; I have hosted it at <https://github.com/oliviergimenez/introduction-to-bayesian-statistics-with-R> and will update it. Practicing helps to better understand and to check that we have indeed understood. If you are reading the electronic version available at <https://oliviergimenez.github.io/introduction-to-bayesian-statistics-with-R>, you can copy the lines of code and then paste them into `R` to run them. To save some space and avoid disrupting the reading too much, some code is not shown, in particular the code used to produce the figures, but it is available at <https://github.com/oliviergimenez/introduction-to-bayesian-statistics-with-R>. There you will find (i) the complete `R` code and the texts that make up the chapters of this book (the following `R Markdown` files: `index.Rmd`, `01-principles.Rmd`, `02-mcmcmethods.Rmd`, `03-implementation.Rmd`, `04-priors.Rmd`, `05-regression.Rmd`, `06-glms.Rmd` and `07-conclusions.Rmd`) as well as (ii) the `R` scripts cleaned of the text to allow you to run the code more easily (the compressed file `scriptsR.zip`).

## Further reading {.unnumbered}

If you would like to go further, I recommend the following books, whose list is of course not exhaustive. These books were a source of inspiration in writing this book. I hesitated to provide more references, and to cite (many) scientific articles, but I will not do so; the books below are more than sufficient.

-   Bayesian Methods for Ecology [@mccarthy2007]. A short and truly accessible book to understand how to apply Bayesian statistics in ecology without getting lost in the mathematics. The book website is here <https://bit.ly/4jSlfQL>.

-   Applied Statistical Modelling for Ecologists: A Practical Guide to Bayesian and Likelihood Inference Using R, JAGS, NIMBLE, Stan and TMB [@kery2024]. A practical manual to learn how to model with the main Bayesian tools in R (JAGS, NIMBLE, Stan or TMB), based on concrete ecological examples and comparisons of results. The companion website with the code is here <https://www.elsevier.com/books-and-journals/book-companion/9780443137150>.

-   Bayes Rules!: An Introduction to Applied Bayesian Modeling [@bayesrules2024]. A very pedagogical book to discover the principles and applications of Bayesian statistics in an intuitive and progressive way. The book is available online here <https://www.bayesrulesbook.com/>.

-   Doing Bayesian Data Analysis: A Tutorial with R and Bugs [@kruschke2010]. A thorough and visual tutorial that guides the learning of Bayesian statistics step by step with many practical examples. Everything is available at <https://sites.google.com/site/doingbayesiandataanalysis/>.

-   Bayesian Data Analysis [@gelman2013]. The reference book for those who wish to acquire a solid theoretical and applied understanding of Bayesian statistics. The book website is here <https://sites.stat.columbia.edu/gelman/book/>.

-   Statistical Rethinking: A Bayesian Course with Examples in R and Stan [@mcelreath2020]. A captivating book to learn how to build and interpret Bayesian models by first developing statistical intuition. All the details are here <https://xcelab.net/rm/>, and I highly recommend the video course here <https://github.com/rmcelreath/stat_rethinking_2024>.

## How this book was written {.unnumbered}

This book was written in `RStudio` (<http://www.rstudio.com/ide/>) using the `bookdown` package (<http://bookdown.org/>). The website is hosted via GitHub Pages (<https://pages.github.com/>).



I used `R` version R-4.5.2_2025-10-31 and the following packages:



|package   |version |source         |
|:---------|:-------|:--------------|
|bookdown  |0.43    |CRAN (R 4.5.0) |
|brms      |2.22.0  |CRAN (R 4.5.0) |
|lme4      |1.1-37  |CRAN (R 4.5.0) |
|MCMCvis   |0.16.3  |CRAN (R 4.5.0) |
|nimble    |1.3.0   |CRAN (R 4.5.0) |
|posterior |1.6.1   |CRAN (R 4.5.0) |
|tidyverse |2.0.0   |CRAN (R 4.5.0) |
|visreg    |2.7.0   |CRAN (R 4.5.0) |



## About the author {.unnumbered}

My name is Olivier Gimenez (<https://oliviergimenez.github.io/>). I am a senior scientist at CNRS. After studying mathematics, I completed a PhD in statistics applied to ecology. I later obtained my habilitation (HDR) in ecology and evolution and returned to university to study sociology.

I have authored scientific articles (<https://oliviergimenez.github.io/publication/papers/>) using Bayesian statistics and co-written several books (<https://oliviergimenez.github.io/publication/books/>), some of which also cover Bayesian methods.

You can find me on BlueSky ([oaggimenez.bsky.social](https://bsky.app/profile/oaggimenez.bsky.social)) and LinkedIn ([olivier-gimenez-545451115/](https://www.linkedin.com/in/olivier-gimenez-545451115/)), or contact me by email.

## Acknowledgements {.unnumbered}

I thank my employer, the French National Centre for Scientific Research (CNRS). Being a researcher is a meaningful and valuable profession. However, we are witnessing a gradual deterioration of working conditions in academia, with increased competition, precarity, and fewer permanent positions. I am fortunate to work in a supportive environment at the Centre for Functional and Evolutionary Ecology (CEFE), where collaboration and collective spirit remain strong.

My interest in Bayesian statistics dates back to my postdoctoral years in England and Scotland. I thank Byron Morgan for giving me the freedom to explore this field, Ruth King for our collaborations and my first experience writing a book, and Steve Brooks for the many stimulating discussions.

I am grateful to the Master's students I have taught for over ten years - they have unknowingly contributed to shaping this project. I also thank all students and postdoctoral researchers who have shared part of their journey with me.

Thanks to the people who kindly agreed to read parts of this book.

Finally, this book is dedicated to Eleni, Gabriel, and Mélina.
