# References

* Darling, D. A. and Robbins, H. (1967). Confidence sequences for mean, variance, and median. *Proceedings of the National Academy of Sciences*, 58(1).
* Howard, S. R., Ramdas, A., McAuliffe, J. and Sekhon, J. (2020). Time-uniform Chernoff bounds via nonnegative supermartingales. *Probability Surveys*, 17. [arXiv:1808.03204](https://arxiv.org/abs/1808.03204)
* Howard, S. R., Ramdas, A., McAuliffe, J. and Sekhon, J. (2021). Time-uniform, nonparametric, nonasymptotic confidence sequences. *The Annals of Statistics*, 49(2). [arXiv:1810.08240](https://arxiv.org/abs/1810.08240)
* Howard, S. R. and Ramdas, A. (2022). Sequential estimation of quantiles with applications to A/B-testing and best-arm identification. *Bernoulli*, 28(3). [arXiv:1906.09712](https://arxiv.org/abs/1906.09712)
* Johari, R., Koomen, P., Pekelis, L. and Walsh, D. (2022). Always valid inference: continuous monitoring of A/B tests. *Operations Research*, 70(3). [arXiv:1512.04922](https://arxiv.org/abs/1512.04922)
* Ramdas, A., Grünwald, P., Vovk, V. and Shafer, G. (2023). Game-theoretic statistics and safe anytime-valid inference. *Statistical Science*, 38(4). [arXiv:2210.01948](https://arxiv.org/abs/2210.01948)
* Robbins, H. (1970). Statistical methods related to the law of the iterated logarithm. *The Annals of Mathematical Statistics*, 41(5).
* Shafer, G., Shen, A., Vereshchagin, N. and Vovk, V. (2011). Test martingales, Bayes factors and p-values. *Statistical Science*, 26(1).
* Ville, J. (1939). *Étude critique de la notion de collectif*. Gauthier-Villars, Paris.
* Vovk, V. and Wang, R. (2021). E-values: calibration, combination and applications. *The Annals of Statistics*, 49(3). [arXiv:1912.06116](https://arxiv.org/abs/1912.06116)
* Wang, R. and Ramdas, A. (2022). False discovery rate control with e-values. *Journal of the Royal Statistical Society Series B*, 84(3). [arXiv:2009.02824](https://arxiv.org/abs/2009.02824)
* Waudby-Smith, I. and Ramdas, A. (2024). Estimating means of bounded random variables by betting. *Journal of the Royal Statistical Society Series B*, 86(1). [arXiv:2010.09686](https://arxiv.org/abs/2010.09686)
* Waudby-Smith, I., Arbour, D., Sinha, R., Kennedy, E. H. and Ramdas, A. (2024). Time-uniform central limit theory and asymptotic confidence sequences. *The Annals of Statistics*, 52(6). [arXiv:2103.06476](https://arxiv.org/abs/2103.06476)

## Related software

Other packages implement parts of this area. [confseq](https://github.com/gostevehoward/confseq) by Steve Howard provides uniform boundaries and confidence sequences in C++ with Python bindings. [expectation](https://github.com/jakorostami/expectation) and `savvi` are Python packages for e-values and safe anytime valid inference, and [safestats](https://cran.r-project.org/package=safestats) implements safe tests in R. `ville` provides a common interface across means, proportions, quantiles, variances and A/B tests, states the source of each estimator, and tests coverage by simulation.

## Citing ville

Use the "Cite this repository" button on [GitHub](https://github.com/shivsbhaduri/ville), or the metadata in [`CITATION.cff`](https://github.com/shivsbhaduri/ville/blob/main/CITATION.cff).
