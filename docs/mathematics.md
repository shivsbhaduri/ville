# Mathematics

Notation: $X_1, X_2, \dots$ for the stream, $S_t = \sum_{i \le t} (X_i - \mu)$ for the centred running sum and $\bar X_t$ for the running mean.

## Test martingales and Ville's inequality

A **test martingale** for a null hypothesis $H_0$ is a process $M_0 = 1, M_1, M_2, \dots$ that is nonnegative and, under $H_0$, a supermartingale: $\mathbb{E}[M_t \mid X_1, \dots, X_{t-1}] \le M_{t-1}$. Equivalently, $M_t$ is the wealth of a gambler who starts with one unit and bets against $H_0$ at fair odds.

**Ville's inequality** (Ville 1939):

$$
\Pr_{H_0}\left( \exists\, t \ge 1 : M_t \ge \frac{1}{\alpha} \right) \le \alpha .
$$

Proof: let $\tau$ be the first $t$ with $M_t \ge 1/\alpha$. By optional stopping $\mathbb{E}[M_{\tau \wedge t}] \le 1$, hence $\Pr(\tau \le t) / \alpha \le 1$ for every $t$; let $t \to \infty$.

The inequality is a time uniform form of Markov's inequality. It implies that the test rejecting $H_0$ at the first $t$ with $M_t \ge 1/\alpha$ has level $\alpha$ under any stopping rule. Shafer et al. (2011) and Ramdas et al. (2023) survey the area.

## E-values, e-processes and anytime valid p-values

An **e-value** is a nonnegative statistic $E$ with $\mathbb{E}_{H_0}[E] \le 1$. Large values are evidence against $H_0$, and $\Pr(E \ge 1/\alpha) \le \alpha$ by Markov's inequality. An **e-process** is a sequence $E_t$ such that $E_\tau$ is an e-value for every stopping time $\tau$. Test martingales are e-processes.

From an e-process `ville` reports the **anytime valid p-value**

$$
p_t = \min\left(1, \frac{1}{\max_{s \le t} E_s}\right),
$$

which is nonincreasing and satisfies $\Pr_{H_0}(\exists\, t : p_t \le \alpha) \le \alpha$.

## Confidence sequences

A **confidence sequence** is a sequence of sets $C_1, C_2, \dots$ with

$$
\Pr\left( \forall\, t \ge 1 : \theta \in C_t \right) \ge 1 - \alpha .
$$

Confidence sequences were introduced by Darling and Robbins (1967). A confidence interval satisfies only $\Pr(\theta \in C_n) \ge 1 - \alpha$ for a fixed $n$. Confidence sequences are obtained by inverting a family of e-processes indexed by the parameter:

$$
C_t = \{\, \theta_0 : E_t(\theta_0) < 1/\alpha \,\}.
$$

By Ville's inequality the probability that the true $\theta$ is ever excluded is at most $\alpha$. The running intersection $\bigcap_{s \le t} C_s$ is therefore also a confidence sequence; `ville` returns it by default (`intersect=True`).

Time uniform coverage increases width. By the law of the iterated logarithm no confidence sequence can shrink faster than order $\sqrt{\log\log t / t}$, against $\sqrt{1/t}$ for a fixed $n$ interval. In the [benchmarks](benchmarks.md) the variance adaptive sequences are 1.4 to 2 times the width of the fixed $n$ CLT interval.

## Normal mixtures: `GaussianMixtureCS` and `MSPRT`

Suppose the observations are $\sigma$ sub-Gaussian, meaning $\mathbb{E}[e^{\lambda (X - \mu)}] \le e^{\lambda^2 \sigma^2 / 2}$ for all $\lambda$. Then for each fixed $\lambda$

$$
M_t(\lambda) = \exp\left( \lambda S_t - \tfrac{1}{2} \lambda^2 \sigma^2 t \right)
$$

is a test martingale for the true mean. A single $\lambda$ has power against one effect size only. Robbins (1970) mixes over $\lambda \sim N(0, 1/\rho)$, which gives the closed form

$$
M_t = \int M_t(\lambda)\, dF(\lambda) = \sqrt{\frac{\rho}{\sigma^2 t + \rho}}\; \exp\left( \frac{S_t^2}{2(\sigma^2 t + \rho)} \right).
$$

Setting $M_t = 1/\alpha$ and solving for $S_t$ gives the confidence sequence

$$
\bar X_t \pm \frac{1}{t} \sqrt{ (\sigma^2 t + \rho) \log\left( \frac{\sigma^2 t + \rho}{\rho\, \alpha^2} \right) } .
$$

The width is of order $\sqrt{\log t / t}$. The mixing precision $\rho$ determines the sample size at which the boundary is tightest. Minimising the boundary at $t^\star$ gives $r - \log r = 1 + 2\log(1/\alpha)$ with $r = (\sigma^2 t^\star + \rho)/\rho$; `ville` solves this numerically for the `t_opt` argument. The choice of $\rho$ affects width only.

By Hoeffding's lemma, data on $[a, b]$ are $(b - a)/2$ sub-Gaussian, so the sequence applies to bounded data. It does not adapt to the variance and is the widest method in the package when the variance is small.

`MSPRT` is the corresponding test. Johari et al. (2022) place a $N(\theta_0, \tau^2)$ prior on the alternative mean, which is the mixture above with $\rho = \sigma^4 / \tau^2$:

$$
\Lambda_t = \sqrt{\frac{\sigma^2}{\sigma^2 + t \tau^2}}\; \exp\left( \frac{t^2 \tau^2 (\bar X_t - \theta_0)^2}{2 \sigma^2 (\sigma^2 + t \tau^2)} \right).
$$

If `tau` is not given it is set from `t_opt` as above.

## Asymptotic confidence sequences: `AsymptoticCS`

When there are no known bounds and no known scale, Waudby-Smith et al. (2024) show that the normal mixture boundary with $\sigma$ replaced by the sample standard deviation $\hat\sigma_t$ gives an **asymptotic confidence sequence**:

$$
\bar X_t \pm \hat\sigma_t \sqrt{ \frac{2 (t \rho^2 + 1)}{t^2 \rho^2} \log\left( \frac{\sqrt{t \rho^2 + 1}}{\alpha} \right) } .
$$

$\rho$ is set from `t_opt` as above. The sequence is the time uniform analogue of the CLT interval $\bar X_t \pm z_{1-\alpha/2}\, \hat\sigma_t / \sqrt t$ and requires finite moments of order slightly above two. The guarantee is asymptotic: in the definition of Waudby-Smith et al. the sequence is coupled to a nonasymptotic one with width ratio tending to one, and coverage from time $m$ onwards tends to at least $1 - \alpha$ as $m \to \infty$. `ville` therefore returns an infinite interval until `min_samples` observations (default 100) are available. The coverage tests check this default on exponential data.

## Betting confidence sequences: `BettingCS` and `BettingTest`

For data in $[0, 1]$ (other known ranges are rescaled) and a candidate mean $m$, define the capital process

$$
K_t(m) = \prod_{i=1}^{t} \bigl( 1 + \lambda_i (X_i - m) \bigr).
$$

If $\lambda_i$ is predictable (a function of $X_1, \dots, X_{i-1}$) and lies in $[-1/(1-m),\, 1/m]$, then $K_t(m) \ge 0$ and, when the mean is $m$, $K_t(m)$ is a martingale. This holds for every distribution on $[0, 1]$ with mean $m$. No parametric model is required, and the predictable $\lambda_i$ adapt to the variance.

Waudby-Smith and Ramdas (2024, Theorem 3) use the hedged process

$$
K_t^{\pm}(m) = \max\Bigl\{ \theta \prod_{i \le t} \bigl(1 + \lambda_i^{+}(m)(X_i - m)\bigr),\;
(1 - \theta) \prod_{i \le t} \bigl(1 - \lambda_i^{-}(m)(X_i - m)\bigr) \Bigr\},
$$

and $C_t = \{ m : K_t^{\pm}(m) < 1/\alpha \}$, which is an interval. The bets follow their predictable plug-in rule

$$
\lambda_i^{+}(m) = \min\left( \dot\lambda_i,\, \frac{c}{m} \right), \quad
\lambda_i^{-}(m) = \min\left( \dot\lambda_i,\, \frac{c}{1 - m} \right), \quad
\dot\lambda_i = \sqrt{ \frac{2 \log(2/\alpha)}{\hat\sigma_{i-1}^2\, i \log(1 + i)} },
$$

with $c = 1/2$ by default and the regularised running estimates

$$
\hat\mu_t = \frac{1/2 + \sum_{i \le t} X_i}{t + 1}, \qquad
\hat\sigma_t^2 = \frac{1/4 + \sum_{i \le t} (X_i - \hat\mu_i)^2}{t + 1}.
$$

`ville` evaluates $K_t^{\pm}$ on a grid of candidate means (`grid=10_000` by default) and widens the reported interval to the adjacent grid points, so the discretisation is conservative. Under the running intersection, excluded candidates cannot re-enter, so only grid points inside the current interval are updated and the cost per observation decreases as the interval shrinks.

`BettingTest` evaluates the capital process at a single null value $m_0$. The two sided e-process is $\theta K^{+} + (1-\theta) K^{-}$, a martingale under $H_0$. The one sided tests use $K^{+}$ or $K^{-}$ alone and are also valid for the composite nulls $\mu \le m_0$ and $\mu \ge m_0$.

## Empirical Bernstein: `EmpiricalBernsteinCS`

A closed form sequence from the same paper (Waudby-Smith and Ramdas 2024, Theorem 2). With $\lambda_i = \min(\dot\lambda_i, c)$, $v_i = 4 (X_i - \hat\mu_{i-1})^2$ and $\psi_E(\lambda) = (-\log(1 - \lambda) - \lambda)/4$,

$$
C_t = \left( \frac{\sum_{i \le t} \lambda_i X_i}{\sum_{i \le t} \lambda_i} \;\pm\; \frac{\log(2/\alpha) + \sum_{i \le t} v_i\, \psi_E(\lambda_i)}{\sum_{i \le t} \lambda_i} \right) \cap [0, 1].
$$

It adapts to the variance and costs $O(1)$ per observation.

## Proportions: `BernoulliCS` and `BernoulliTest`

For binary data with $S_t$ successes in $t$ trials, the Bayes factor of a $\text{Beta}(a, b)$ mixture against the point null $p = p_0$ is

$$
E_t(p_0) = \frac{B(a + S_t,\, b + t - S_t) / B(a, b)}{p_0^{S_t} (1 - p_0)^{t - S_t}} .
$$

The numerator is the marginal likelihood under the mixture and the denominator the likelihood under $H_0$, so $E_t$ is a likelihood ratio with $\mathbb{E}_{H_0}[E_t] = 1$ for all $t$, and hence a test martingale. The test suite verifies the unit mean by exact summation over the binomial distribution. Monte Carlo is unsuitable here because the mean is dominated by sequences of probability around $2^{-200}$.

`BernoulliCS` inverts this family. Since $-\log E_t(p_0)$ is concave in $p_0$ with maximum at $\hat p = S_t / t$, the set $\{ p_0 : E_t(p_0) < 1/\alpha \}$ is an interval; its end points are computed by bisection. The prior affects width only.

## Quantiles: `QuantileCS`

Howard and Ramdas (2022) construct quantile confidence sequences from a counting argument. Let $Q$ be the true $q$ quantile. The number of observations at or below $Q$ is a sum of Bernoulli variables with success probability at least $q$, and the number strictly below $Q$ a sum with probability at most $q$. A time uniform boundary $u_t$ for a centred sum of Bernoulli($q$) variables therefore gives, with probability at least $1 - \alpha$ for all $t$ at once,

$$
X_{(\lceil t q - u_t \rceil)} \;\le\; Q \;\le\; X_{(\lfloor t q + u_t \rfloor + 1)} ,
$$

where $X_{(k)}$ is the $k$-th order statistic of the first $t$ observations, and the bound is infinite when the index falls outside $1, \dots, t$. No distributional assumption is needed beyond i.i.d. sampling. Ties and atoms are permitted; the tests include a distribution with half its mass at the median.

`ville` uses for $u_t$ the one sided normal mixture boundary at level $\alpha/2$ per side, with variance proxy $1/4$ (Hoeffding). The half normal mixture over $\lambda \ge 0$ gives

$$
M_t = 2 \sqrt{\frac{\rho}{v + \rho}}\; \exp\left( \frac{s^2}{2(v + \rho)} \right) \Phi\left( \frac{s}{\sqrt{v + \rho}} \right), \qquad v = t/4,
$$

and $u_t$ solves $M_t = 2/\alpha$, computed by bisection.

## Variances: `VarianceCS`

For i.i.d. data on $[a, b]$, consecutive pairs give $Y_k = (X_{2k} - X_{2k-1})^2 / 2$, which has mean $\sigma^2$ and lies in $[0, (b - a)^2 / 2]$. A betting confidence sequence for the mean of the $Y_k$ is therefore a confidence sequence for $\sigma^2$. It is capped at $(b - a)^2/4$, the maximum variance on $[a, b]$. The construction is exactly valid but uses half the information of a full U-statistic, so it is conservative.

## A/B tests: `AsymptoticABTest` and `BettingABTest`

In a randomised experiment unit $i$ is treated ($A_i = 1$) with known probability $\pi$. For any centre $m_{i-1}$ computed from earlier units only, the pseudo outcome

$$
\phi_i = (Y_i - m_{i-1}) \left( \frac{A_i}{\pi} - \frac{1 - A_i}{1 - \pi} \right)
$$

has conditional mean $\mathbb{E}[Y(1)] - \mathbb{E}[Y(0)]$, the average treatment effect, because the centre contributes $m_{i-1} - m_{i-1} = 0$. A confidence sequence for the mean of the $\phi_i$ is therefore a confidence sequence for the effect. `ville` uses the running mean of past outcomes as the centre. Without centring, the inverse propensity weights carry the baseline level of the outcome into the variance of $\phi_i$, which is large for metrics such as revenue whose mean is far from zero.

`AsymptoticABTest` applies `AsymptoticCS` to the $\phi_i$, following the randomised experiment setting of Waudby-Smith et al. (2024). `BettingABTest` applies `BettingCS`, using the bound $|\phi_i| \le (b - a) / \min(\pi, 1 - \pi)$ for outcomes in $[a, b]$. It is nonasymptotic.

## Working with e-values: `ville.evalues`

**Calibration.** Any p-value can be turned into an e-value with a calibrator (Vovk and Wang 2021). For $\kappa \in (0, 1)$, $\kappa p^{\kappa - 1}$ is one. Integrating over $\kappa$ gives

$$
\int_0^1 \kappa p^{\kappa - 1}\, d\kappa = \frac{1 - p + p \log p}{p (\log p)^2},
$$

which is what `p_to_e(p)` returns. In the other direction `e_to_p(e)` returns $\min(1, 1/e)$.

**Combination.** The average of e-values for the same null is an e-value under any dependence. The product is an e-value when they are independent, or when each is valid conditionally on its predecessors, as with e-values from successive experiments.

**Multiple testing.** For $m$ hypotheses with e-values $e_1, \dots, e_m$, e-BH (Wang and Ramdas 2022) rejects the $\hat k$ hypotheses with the largest e-values, where

$$
\hat k = \max\left\{ k : \frac{k\, e_{[k]}}{m} \ge \frac{1}{\alpha} \right\}
$$

and $e_{[k]}$ is the $k$-th largest. It controls the false discovery rate at $\alpha$ under arbitrary dependence between the e-values, as arises when several metrics are computed from one experiment.
