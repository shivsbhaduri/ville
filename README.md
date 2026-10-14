# ville

**Anytime valid inference for Python.** Confidence sequences, sequential tests and e-values that remain valid however often you look at the data and whenever you decide to stop.

A classical confidence interval or p-value carries its guarantee only if the sample size was fixed in advance and the data are analysed once. Experiments are rarely run that way. Dashboards refresh hourly, results are checked daily, and tests are stopped when they look decisive. Each look is another chance for noise to cross the significance line, and the error rate that was promised at 5% quietly becomes 20% or 30%.

The methods in `ville` give guarantees that hold **simultaneously at every sample size**. You can monitor continuously, stop as soon as the evidence is clear, or keep going if it is not, and the error rate stays where you set it. The library is named after Jean Ville, whose 1939 inequality is the result every method here rests on.

![Daily peeking at a z test against a confidence sequence](docs/figures/peeking.png)

*4,000 simulated A/B tests with no true effect, each checked once a day for 30 days. A z test at 5% declares a winner in 29% of them by the end. `ville.AsymptoticABTest`, checked at the same times, stays well under 5%. Script: [`benchmarks/peeking.py`](benchmarks/peeking.py).*

## Contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [What is in the library](#what-is-in-the-library)
4. [The mathematics](#the-mathematics)
5. [Worked examples](#worked-examples)
6. [Benchmarks](#benchmarks)
7. [Choosing a method](#choosing-a-method)
8. [Assumptions and limitations](#assumptions-and-limitations)
9. [Reproducing the results](#reproducing-the-results)
10. [Related software](#related-software)
11. [References](#references)

## Installation

`ville` is not on PyPI yet. Install it from a clone:

```bash
git clone https://github.com/shivsbhaduri/ville.git
cd ville
pip install -e .
```

It needs Python 3.10 or later, NumPy and SciPy. Matplotlib is only needed to redraw the figures.

## Quick start

Every confidence sequence has the same interface. `fit` takes a whole stream, `update` adds new observations to it, and the running results sit in `estimate_`, `lower_` and `upper_`, one entry per observation.

```python
import numpy as np
import ville

rng = np.random.default_rng(42)
n = 40_000
treated = (rng.random(n) < 0.5).astype(float)
converted = (rng.random(n) < np.where(treated == 1, 0.050, 0.040)).astype(float)

control = ville.BernoulliCS(alpha=0.05).fit(converted[treated == 0])
test = ville.BettingABTest(bounds=(0, 1), alpha=0.05).fit(converted, treated)

for t in (5000, 10_000, 20_000, n):
    print(t, test.estimate_[t - 1], test.lower_[t - 1], test.upper_[t - 1])
print(test.stopping_time())
```

Output from [`examples/conversion_rate.py`](examples/conversion_rate.py):

```text
Control conversion rate
  after    500 users: 0.0480  [0.0214, 0.0699]
  after  2,000 users: 0.0410  [0.0277, 0.0605]
  after 10,000 users: 0.0408  [0.0335, 0.0492]
  after 20,147 users: 0.0399  [0.0348, 0.0459]

Difference in conversion rate, treatment minus control
  after  5,000 users: +0.0126  [-0.0073, +0.0331]
  after 10,000 users: +0.0118  [-0.0025, +0.0271]
  after 20,000 users: +0.0101  [-0.0001, +0.0223]
  after 40,000 users: +0.0098  [+0.0019, +0.0189]

Zero first excluded after 20,579 users (betting)
Zero first excluded after 16,532 users (asymptotic)
```

Every row of that output is valid at the same time. Reading the interval after 5,000 users, deciding to continue, and reading it again after 20,000 does not cost any coverage. The true lift here is 0.010, and every interval printed contains it.

Streams can also arrive in pieces:

```python
cs = ville.EmpiricalBernsteinCS(bounds=(0, 1))
for batch in daily_batches:
    cs.update(batch)
    lo, hi = cs.interval()
```

## What is in the library

| Class | Target | Assumptions | Method |
|---|---|---|---|
| `GaussianMixtureCS` | mean | sub-Gaussian with known scale | Robbins (1970); Howard et al. (2021) |
| `AsymptoticCS` | mean | finite variance, large sample | Waudby-Smith et al. (2024) |
| `EmpiricalBernsteinCS` | mean | known bounds | Waudby-Smith and Ramdas (2024), Theorem 2 |
| `BettingCS` | mean | known bounds | Waudby-Smith and Ramdas (2024), Theorem 3 |
| `BernoulliCS` | proportion | binary data | beta-binomial mixture; Howard et al. (2021) |
| `QuantileCS` | any quantile | none beyond i.i.d. | Howard and Ramdas (2022) |
| `VarianceCS` | variance | known bounds | paired differences with `BettingCS` |
| `AsymptoticABTest` | average treatment effect | randomised, finite variance | Waudby-Smith et al. (2024) |
| `BettingABTest` | average treatment effect | randomised, bounded outcome | Waudby-Smith and Ramdas (2024) |
| `MSPRT` | Gaussian mean test | Gaussian or sub-Gaussian | Johari et al. (2022) |
| `BettingTest` | bounded mean test | known bounds | Waudby-Smith and Ramdas (2024) |
| `BernoulliTest` | proportion test | binary data | beta-binomial Bayes factor |
| `evalues.p_to_e`, `e_to_p`, `combine` | e-value calculus | see below | Vovk and Wang (2021) |
| `evalues.ebh` | false discovery rate | arbitrary dependence | Wang and Ramdas (2022) |
| `simulate.peeking`, `simulate.stopping_times` | diagnostics | | |

Confidence sequences inherit from `ville.ConfidenceSequence` and sequential tests from `ville.SequentialTest`. Tests expose `e_values_`, anytime valid `p_values_` and `stopping_time(alpha)`.

## The mathematics

This section sets out what each estimator computes and why it is valid. The notation is $X_1, X_2, \dots$ for the stream, $S_t = \sum_{i \le t} (X_i - \mu)$ for the centred running sum and $\bar X_t$ for the running mean.

### Test martingales and Ville's inequality

A **test martingale** for a null hypothesis $H_0$ is a process $M_0 = 1, M_1, M_2, \dots$ that is nonnegative and, under $H_0$, a supermartingale: $\mathbb{E}[M_t \mid X_1, \dots, X_{t-1}] \le M_{t-1}$. It can be read as the wealth of a gambler who starts with one unit and bets against $H_0$ at fair odds. If $H_0$ is true the gambler cannot expect to make money.

**Ville's inequality** (Ville 1939) says that such a process rarely gets large at any point:

```math
\Pr_{H_0}\left( \exists\, t \ge 1 : M_t \ge \frac{1}{\alpha} \right) \le \alpha .
```

The proof is two lines. Let $\tau$ be the first time $M_t \ge 1/\alpha$. Optional stopping gives $\mathbb{E}[M_{\tau \wedge t}] \le 1$ for every $t$, so $\Pr(\tau \le t) / \alpha \le \mathbb{E}[M_{\tau \wedge t}] \le 1$, and letting $t \to \infty$ finishes it.

This is Markov's inequality made uniform over time, and it is the source of everything else. Shafer et al. (2011) and Ramdas et al. (2023) give fuller accounts of testing by betting. Rejecting $H_0$ the first time $M_t \ge 1/\alpha$ is a level $\alpha$ test whichever stopping rule is used, including "stop when it looks good".

### E-values, e-processes and anytime valid p-values

An **e-value** is a nonnegative statistic $E$ with $\mathbb{E}_{H_0}[E] \le 1$. Large values are evidence against $H_0$, and $\Pr(E \ge 1/\alpha) \le \alpha$ by Markov's inequality. An **e-process** is a sequence $E_t$ such that $E_\tau$ is an e-value for every stopping time $\tau$. A test martingale is the main example.

From an e-process `ville` reports the **anytime valid p-value**

```math
p_t = \min\left(1, \frac{1}{\max_{s \le t} E_s}\right),
```

which is nonincreasing and satisfies $\Pr_{H_0}(\exists\, t : p_t \le \alpha) \le \alpha$.

### Confidence sequences

A **confidence sequence** is a sequence of sets $C_1, C_2, \dots$ with

```math
\Pr\left( \forall\, t \ge 1 : \theta \in C_t \right) \ge 1 - \alpha .
```

The idea goes back to Darling and Robbins (1967). Compare the confidence interval, which promises only $\Pr(\theta \in C_n) \ge 1 - \alpha$ for one fixed $n$. Confidence sequences come from inverting a family of e-processes, one for each candidate value:

```math
C_t = \{\, \theta_0 : E_t(\theta_0) < 1/\alpha \,\}.
```

By Ville's inequality the true $\theta$ is excluded at some time with probability at most $\alpha$. Because the guarantee covers all times at once, the running intersection $\bigcap_{s \le t} C_s$ is also a confidence sequence. `ville` reports it by default (`intersect=True`), which keeps the intervals nested.

Time uniform coverage has a price in width. The law of the iterated logarithm shows that no confidence sequence can shrink faster than order $\sqrt{\log\log t / t}$, compared with $\sqrt{1/t}$ for a fixed $n$ interval. In practice the variance adaptive methods below are about 1.4 to 2 times as wide as a fixed $n$ interval (see [Benchmarks](#benchmarks)).

### Normal mixtures: `GaussianMixtureCS` and `MSPRT`

Suppose the observations are $\sigma$ sub-Gaussian, meaning $\mathbb{E}[e^{\lambda (X - \mu)}] \le e^{\lambda^2 \sigma^2 / 2}$ for all $\lambda$. Then for each fixed $\lambda$

```math
M_t(\lambda) = \exp\left( \lambda S_t - \tfrac{1}{2} \lambda^2 \sigma^2 t \right)
```

is a test martingale for the true mean. A single $\lambda$ is only powerful for one effect size, so Robbins (1970) averaged over $\lambda \sim N(0, 1/\rho)$. The integral is explicit:

```math
M_t = \int M_t(\lambda)\, dF(\lambda) = \sqrt{\frac{\rho}{\sigma^2 t + \rho}}\; \exp\left( \frac{S_t^2}{2(\sigma^2 t + \rho)} \right).
```

Setting $M_t = 1/\alpha$ and solving for $S_t$ gives the confidence sequence

```math
\bar X_t \pm \frac{1}{t} \sqrt{ (\sigma^2 t + \rho) \log\left( \frac{\sigma^2 t + \rho}{\rho\, \alpha^2} \right) } .
```

The width shrinks like $\sqrt{\log t / t}$. The mixing precision $\rho$ decides where the sequence is tightest. Minimising the boundary at a chosen sample size $t^\star$ gives the condition $r - \log r = 1 + 2\log(1/\alpha)$ with $r = (\sigma^2 t^\star + \rho)/\rho$, which `ville` solves numerically. That is the `t_opt` argument. Validity does not depend on it.

Bounded data on $[a, b]$ are $(b - a)/2$ sub-Gaussian by Hoeffding's lemma, so this sequence works for them too, though it ignores the variance and is the widest method in the library on low variance data.

`MSPRT` is the same object written as a test. Johari et al. (2022) place a $N(\theta_0, \tau^2)$ prior on the alternative mean, which is the mixture above with $\rho = \sigma^4 / \tau^2$:

```math
\Lambda_t = \sqrt{\frac{\sigma^2}{\sigma^2 + t \tau^2}}\; \exp\left( \frac{t^2 \tau^2 (\bar X_t - \theta_0)^2}{2 \sigma^2 (\sigma^2 + t \tau^2)} \right).
```

If `tau` is not given it is set from `t_opt` as above.

### Asymptotic confidence sequences: `AsymptoticCS`

When there are no known bounds and no known scale, Waudby-Smith et al. (2024) show that the normal mixture boundary with $\sigma$ replaced by the sample standard deviation $\hat\sigma_t$ gives an **asymptotic confidence sequence**:

```math
\bar X_t \pm \hat\sigma_t \sqrt{ \frac{2 (t \rho^2 + 1)}{t^2 \rho^2} \log\left( \frac{\sqrt{t \rho^2 + 1}}{\alpha} \right) } .
```

Here $\rho$ is chosen from `t_opt` in the same way as before. This is the time uniform analogue of the CLT interval $\bar X_t \pm z_{1-\alpha/2}\, \hat\sigma_t / \sqrt t$. It needs only a finite variance (formally, slightly more than two moments). Like the CLT interval its guarantee is a large sample one. In their definition the sequence is coupled to an exact one with the same width in the limit, and coverage from time $m$ onwards tends to at least $1 - \alpha$ as $m$ grows. For that reason `ville` reports an infinite interval until `min_samples` observations (default 100) have arrived. The coverage tests use exponential data, which are strongly skewed, to check this setting.

### Betting confidence sequences: `BettingCS` and `BettingTest`

For data in $[0, 1]$ (any known range is rescaled to it) and a candidate mean $m$, consider the wealth of a gambler who, at each step, stakes a fraction $\lambda_i$ of their wealth on $X_i$ exceeding $m$:

```math
K_t(m) = \prod_{i=1}^{t} \bigl( 1 + \lambda_i (X_i - m) \bigr).
```

If $\lambda_i$ is chosen using only $X_1, \dots, X_{i-1}$ and lies in $[-1/(1-m),\, 1/m]$, wealth stays nonnegative and, when the true mean is $m$, $K_t(m)$ is a martingale. This holds for **every** distribution on $[0, 1]$ with mean $m$. No parametric family is assumed, and the bets themselves learn the variance.

Waudby-Smith and Ramdas (2024, Theorem 3) split the capital between an upward and a downward gambler,

```math
K_t^{\pm}(m) = \max\Bigl\{ \theta \prod_{i \le t} \bigl(1 + \lambda_i^{+}(m)(X_i - m)\bigr),\;
(1 - \theta) \prod_{i \le t} \bigl(1 - \lambda_i^{-}(m)(X_i - m)\bigr) \Bigr\},
```

and report $C_t = \{ m : K_t^{\pm}(m) < 1/\alpha \}$, which is an interval. The bets follow their predictable plug-in rule,

```math
\lambda_i^{+}(m) = \min\left( \dot\lambda_i,\, \frac{c}{m} \right), \quad
\lambda_i^{-}(m) = \min\left( \dot\lambda_i,\, \frac{c}{1 - m} \right), \quad
\dot\lambda_i = \sqrt{ \frac{2 \log(2/\alpha)}{\hat\sigma_{i-1}^2\, i \log(1 + i)} },
```

with $c = 1/2$ by default and the regularised running estimates

```math
\hat\mu_t = \frac{1/2 + \sum_{i \le t} X_i}{t + 1}, \qquad
\hat\sigma_t^2 = \frac{1/4 + \sum_{i \le t} (X_i - \hat\mu_i)^2}{t + 1}.
```

`ville` evaluates the wealth on a grid of candidate means (`grid=10_000` by default). The reported interval is widened to the neighbouring grid points, so the discretisation can only make it more conservative. With the running intersection on, a candidate that has been ruled out can never come back, so only the grid points still inside the interval are updated. The cost per observation falls as the interval shrinks.

`BettingTest` uses the same wealth at a single null value $m_0$. The two sided test is the average $\theta K^{+} + (1-\theta) K^{-}$, which is a martingale under $H_0$. The one sided tests keep one gambler, which also makes them valid for the composite nulls $\mu \le m_0$ and $\mu \ge m_0$.

### Empirical Bernstein: `EmpiricalBernsteinCS`

A closed form relative of the betting sequence (Waudby-Smith and Ramdas 2024, Theorem 2). With $\lambda_i = \min(\dot\lambda_i, c)$, $v_i = 4 (X_i - \hat\mu_{i-1})^2$ and $\psi_E(\lambda) = (-\log(1 - \lambda) - \lambda)/4$,

```math
C_t = \left( \frac{\sum_{i \le t} \lambda_i X_i}{\sum_{i \le t} \lambda_i} \;\pm\; \frac{\log(2/\alpha) + \sum_{i \le t} v_i\, \psi_E(\lambda_i)}{\sum_{i \le t} \lambda_i} \right) \cap [0, 1].
```

It adapts to the variance like the betting sequence and costs $O(1)$ per observation.

### Proportions: `BernoulliCS` and `BernoulliTest`

For binary data with $S_t$ successes in $t$ trials, the Bayes factor of a $\text{Beta}(a, b)$ mixture against the point null $p = p_0$ is

```math
E_t(p_0) = \frac{B(a + S_t,\, b + t - S_t) / B(a, b)}{p_0^{S_t} (1 - p_0)^{t - S_t}} .
```

The numerator is the probability of the data under the mixture and the denominator its probability under $H_0$, so $\mathbb{E}_{H_0}[E_t] = 1$ at every $t$ and $E_t$ is a test martingale. The test suite checks this exactly by summing over the binomial law, since Monte Carlo cannot see it: the mean is carried by sequences with probability around $2^{-200}$.

`BernoulliCS` inverts this test. As a function of $p_0$, $-\log E_t(p_0)$ is concave with its maximum at $\hat p = S_t / t$, so the set $\{ p_0 : E_t(p_0) < 1/\alpha \}$ is an interval, and `ville` finds its two ends by bisection. The prior affects only the width, never the validity.

### Quantiles: `QuantileCS`

Howard and Ramdas (2022) build confidence sequences for quantiles from a counting argument. Let $Q$ be the true $q$ quantile. The number of observations at or below $Q$ is a sum of Bernoulli variables with success probability at least $q$, and the number strictly below $Q$ a sum with probability at most $q$. A time uniform boundary $u_t$ for a centred sum of Bernoulli($q$) variables therefore gives, with probability at least $1 - \alpha$ for all $t$ at once,

```math
X_{(\lceil t q - u_t \rceil)} \;\le\; Q \;\le\; X_{(\lfloor t q + u_t \rfloor + 1)} ,
```

where $X_{(k)}$ is the $k$-th smallest of the first $t$ observations (and the bound is infinite when the index falls outside $1, \dots, t$). Nothing is assumed about the distribution. Ties and atoms are allowed, and the test suite checks a distribution with half its mass at the median.

`ville` takes $u_t$ to be the one sided normal mixture boundary at level $\alpha/2$ for each side, with the Hoeffding variance $1/4$. With a half normal mixture over $\lambda \ge 0$, the martingale is

```math
M_t = 2 \sqrt{\frac{\rho}{v + \rho}}\; \exp\left( \frac{s^2}{2(v + \rho)} \right) \Phi\left( \frac{s}{\sqrt{v + \rho}} \right), \qquad v = t/4,
```

and $u_t$ is the value of $s$ that makes it equal $2/\alpha$, found by bisection.

### Variances: `VarianceCS`

For i.i.d. data on $[a, b]$, consecutive pairs give $Y_k = (X_{2k} - X_{2k-1})^2 / 2$, which has mean $\sigma^2$ and lies in $[0, (b - a)^2 / 2]$. A betting confidence sequence for the mean of the $Y_k$ is therefore one for $\sigma^2$, and it is capped at $(b - a)^2/4$, the largest variance possible on $[a, b]$. The construction is simple and exactly valid. It uses half the information a full U-statistic would, so it is wider than it needs to be.

### A/B tests: `AsymptoticABTest` and `BettingABTest`

In a randomised experiment unit $i$ is treated ($A_i = 1$) with known probability $\pi$. For any centre $m_{i-1}$ computed from earlier units only, the pseudo outcome

```math
\phi_i = (Y_i - m_{i-1}) \left( \frac{A_i}{\pi} - \frac{1 - A_i}{1 - \pi} \right)
```

has conditional mean $\mathbb{E}[Y(1)] - \mathbb{E}[Y(0)]$, the average treatment effect, because the centre contributes $m_{i-1} - m_{i-1} = 0$. A confidence sequence for the mean of the $\phi_i$ is then one for the effect. `ville` centres on the running mean of past outcomes. Without the centre, the inverse propensity weights would multiply the baseline level of the metric into the variance, which is ruinous for something like revenue with a mean far from zero.

`AsymptoticABTest` applies `AsymptoticCS` to the $\phi_i$, following the randomised experiment setting of Waudby-Smith et al. (2024). `BettingABTest` applies `BettingCS`, using the fact that for outcomes in $[a, b]$ the $\phi_i$ lie in $\pm (b - a) / \min(\pi, 1 - \pi)$. It is valid at every sample size, not only asymptotically.

### Working with e-values: `ville.evalues`

**Calibration.** Any p-value can be turned into an e-value with a calibrator (Vovk and Wang 2021). For $\kappa \in (0, 1)$, $\kappa p^{\kappa - 1}$ is one. Averaging over $\kappa$ removes the choice and gives

```math
\int_0^1 \kappa p^{\kappa - 1}\, d\kappa = \frac{1 - p + p \log p}{p (\log p)^2},
```

which is what `p_to_e(p)` returns. In the other direction `e_to_p(e)` returns $\min(1, 1/e)$.

**Combination.** The average of e-values for the same null is an e-value under any dependence. The product is an e-value when they are independent, or when each is valid conditionally on the earlier ones, which is how evidence from successive experiments can be multiplied.

**Multiple testing.** For $m$ hypotheses with e-values $e_1, \dots, e_m$, e-BH (Wang and Ramdas 2022) rejects the $\hat k$ hypotheses with the largest e-values, where

```math
\hat k = \max\left\{ k : \frac{k\, e_{[k]}}{m} \ge \frac{1}{\alpha} \right\}
```

and $e_{[k]}$ is the $k$-th largest. It controls the false discovery rate at $\alpha$ under **arbitrary dependence** between the e-values, which is the usual situation when many metrics are read off one experiment.

## Worked examples

All examples are in [`examples/`](examples/). The outputs below are from running them as they are.

### Revenue per visitor: an unbounded, skewed metric

Most visitors spend nothing and a few spend a lot. There is no sensible upper bound, so this calls for `AsymptoticABTest`.

```python
rng = np.random.default_rng(7)
n = 100_000
treated = (rng.random(n) < 0.5).astype(float)
buys = rng.random(n) < np.where(treated == 1, 0.10, 0.08)
basket = rng.lognormal(mean=3.5, sigma=0.9, size=n)
revenue = np.where(buys, basket, 0.0)

test = ville.AsymptoticABTest(alpha=0.05, t_opt=20_000).fit(revenue, treated)
stop = test.stopping_time()
```

```text
AsymptoticABTest(n=100000, effect=1.19477, interval=(0.75304, 1.6365))
Stopped after 7,040 visitors
Effect at stopping: +1.464 per visitor, 95% CS [+0.001, +2.928]
True effect: +0.993 per visitor
```

The test stopped after 7,040 of the 100,000 visitors. The point estimate at that moment overstates the effect, as any estimate taken at a data dependent stopping time will (stopping happens when the noise is favourable). The confidence sequence still covers the truth at that moment, and that is the quantity to report.

### Sequential tests and anytime valid p-values

```python
x = rng.normal(0.15, 1.0, 5000)
msprt = ville.MSPRT(theta0=0.0, sigma=1.0, alpha=0.05, t_opt=1000).fit(x)

scores = rng.beta(6, 4, 3000)
bet = ville.BettingTest(m0=0.55, bounds=(0, 1), alternative="greater").fit(scores)
```

```text
MSPRT(n=5000, e=4.899e+24, p=1.399e-25)
  e-value first reached 20 at t = 274
  anytime valid p-value at t =  100: 9.20e-01
  anytime valid p-value at t =  200: 1.25e-01
  anytime valid p-value at t =  300: 1.80e-02
  anytime valid p-value at t = 1000: 2.51e-07
BettingTest(n=3000, e=7.58e+16, p=1.273e-17)
  rejected mean <= 0.55 at t = 59 (true mean 0.6)
Null runs ever rejected at alpha = 0.05: 0.032
```

The last line runs 500 betting tests on data where the null is true and monitors each one at every observation up to 2,000. 3.2% are ever rejected, below the 5% target.

### Latency percentiles

`QuantileCS` makes no assumption about the shape of the distribution, which suits response times.

```python
latency_ms = rng.lognormal(mean=4.0, sigma=0.6, size=10_000)
p95 = ville.QuantileCS(q=0.95, alpha=0.05, t_opt=2000).fit(latency_ms)
```

```text
q = 0.5: true 54.6 ms
  after    200 requests:    55.1 ms  [   46.4,    68.9]
  after  1,000 requests:    56.4 ms  [   52.1,    61.0]
  after 10,000 requests:    55.5 ms  [   54.0,    56.8]
q = 0.95: true 146.5 ms
  after    200 requests:   135.5 ms  [   88.9,     inf]
  after  1,000 requests:   145.1 ms  [  119.2,   313.5]
  after 10,000 requests:   145.6 ms  [  134.1,   161.1]
Variance of U(0, 1), true 0.0833: estimate 0.0832, 95% CS [0.0780, 0.0869]
```

After 200 requests the upper bound for the 95th percentile is still infinite. Ten requests above it is not yet enough to bound a tail without assumptions, and the sequence says so rather than guessing.

### Twenty metrics, correlated, with e-BH

Twenty metrics are read off one experiment, four with real effects. Each user has a shock shared by all their metrics, so the e-values are dependent. e-BH needs no correction for that.

```python
test = ville.MSPRT(theta0=0.0, sigma=np.sqrt(2 * 1.25), t_opt=pairs).fit(y_t - y_c)
...
ville.evalues.ebh(e_values, alpha=0.1)
```

```text
metric  true effect     e-value
     0         0.10    4.86e+20
     1         0.08    3.42e+08
     2         0.06    1.77e+06
     3         0.04    3.48e+05
     4         0.00       0.331
     5         0.00       0.333
     6         0.00        0.33
   ...
Rejected by e-BH at 10%: [0, 1, 2, 3]
Rejected by BH on p = 1/e at 10%: [0, 1, 2, 3]
```

## Benchmarks

Each figure is produced by a script in [`benchmarks/`](benchmarks/) with a fixed seed.

### False positives under peeking

The figure at the top of this page. A two sample z test at 5% is checked once a day for 30 days on experiments with no true effect.

| Day | Users | z test, ever significant | `AsymptoticABTest`, ever significant |
|---:|---:|---:|---:|
| 1 | 333 | 4.9% | 0.1% |
| 2 | 666 | 8.3% | 0.1% |
| 5 | 1,666 | 14.4% | 0.5% |
| 10 | 3,333 | 20.0% | 0.8% |
| 20 | 6,666 | 25.9% | 1.3% |
| 30 | 10,000 | 29.4% | 1.6% |

The confidence sequence would stay under 5% even if it were checked after every single user. Checking daily it is well inside the target.

### Width

![Width of 95% confidence sequences for a bounded mean](docs/figures/widths.png)

Average width over 40 streams of Beta(2, 5) data, which live on $[0, 1]$ with a standard deviation of 0.16.

| n | GaussianMixtureCS | AsymptoticCS | EmpiricalBernsteinCS | BettingCS | CLT interval (fixed n) |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.382 | 0.120 | 0.166 | 0.099 | 0.061 |
| 1,000 | 0.095 | 0.027 | 0.034 | 0.031 | 0.020 |
| 10,000 | 0.032 | 0.009 | 0.012 | 0.012 | 0.006 |
| 20,000 | 0.023 | 0.006 | 0.009 | 0.009 | 0.004 |

Three things stand out. Bounds alone (`GaussianMixtureCS` with the Hoeffding scale) waste a lot of width when the variance is small. The betting and empirical Bernstein sequences recover most of it while staying exactly valid at every $n$, with betting clearly ahead at small samples. The asymptotic sequence is the narrowest of the time uniform methods at large $n$, in exchange for a guarantee that holds only in the limit. The CLT interval is narrower still but is not time uniform, so it is shown for scale only.

### Time to a decision

![When ville.AsymptoticABTest stops on real effects](docs/figures/stopping.png)

A fixed horizon z test sized for 80% power at an effect of 0.05 needs 12,559 users. The confidence sequence is monitored after every user, tuned to the same horizon.

| True effect | Stopped by 12,559 users | Median stopping time | Power of the fixed horizon test |
|---:|---:|---:|---:|
| 0.04 | 34.3% | 17,558 | 61.1% |
| 0.05 | 55.1% | 11,351 | 80.0% |
| 0.08 | 95.6% | 5,013 | 99.4% |

This is the trade being made, stated plainly. At the exact sample size a fixed design was planned for, it has more power than a sequence that has to be valid at every other sample size as well. What the sequence buys is flexibility. Large effects are confirmed in a fraction of the planned sample (a median of 5,013 users against 12,559 when the effect is 0.08). Effects smaller than planned are not lost at the horizon, because the experiment can simply continue. None of this is paid for with false positives.

## Choosing a method

| Your data | Use |
|---|---|
| Binary outcomes (conversion, click, churn) | `BernoulliCS`, `BernoulliTest`; `BettingABTest` for a difference |
| Bounded outcomes with a known range (scores, ratings, capped values) | `BettingCS`; `EmpiricalBernsteinCS` when speed matters |
| Unbounded or heavy tailed (revenue, time spent) | `AsymptoticCS`, `AsymptoticABTest` |
| Gaussian measurements with known noise level | `GaussianMixtureCS`, `MSPRT` |
| Medians, percentiles, latency | `QuantileCS` |
| Spread of a bounded metric | `VarianceCS` |
| Many hypotheses at once | e-values from any of the above, then `evalues.ebh` |

Set `t_opt` near the sample size where you expect to make most decisions. It changes the shape of the sequence, never its validity.

## Assumptions and limitations

* **Independence.** The confidence sequences assume i.i.d. observations, or for the A/B tests units that are independently randomised with a known probability. The betting and normal mixture constructions hold under weaker martingale conditions (a constant conditional mean, for example), but the documentation and tests here cover the i.i.d. case.
* **Bounds must be true bounds.** `BettingCS`, `EmpiricalBernsteinCS`, `VarianceCS` and `BettingABTest` raise an error on any observation outside the stated range. Choosing the range after looking at the data invalidates the guarantee.
* **The asymptotic methods are asymptotic.** `AsymptoticCS` and `AsymptoticABTest` behave like the CLT interval. With heavy skew, raise `min_samples`.
* **Estimates at a stopping time are biased.** Stopping when the evidence is strong selects favourable noise. Report the confidence sequence at the stopping time, not only the point estimate.
* **Grid resolution.** `BettingCS` is computed on a grid of candidate means and widened to the neighbouring grid points. Raise `grid` if the reported interval is close to the grid spacing.
* **`VarianceCS` is simple, not optimal.** It is exactly valid but uses half the available information.
* **Experiment hygiene is out of scope.** Sample ratio mismatch, novelty effects and interference between units break the randomisation assumption, and no statistic can fix that afterwards.

## Reproducing the results

```bash
pip install -e ".[test,examples]"
pytest                                # 41 tests, about 15 seconds
python examples/conversion_rate.py    # and the other examples
cd benchmarks && python peeking.py    # redraws docs/figures/peeking.png
```

The tests check every confidence sequence for time uniform coverage by simulation. A sequence fails a run if it excludes the truth at any time, and the failure rate must stay below $\alpha$. They also check the closed forms against direct calculation and the e-processes for unit mean under the null, confirm that streaming updates give the same answer as a single batch fit, and test e-BH against a hand calculation and in simulation.

## Related software

Other packages cover parts of this area. [confseq](https://github.com/gostevehoward/confseq) by Steve Howard provides uniform boundaries and confidence sequences in C++ with Python bindings. [expectation](https://github.com/jakorostami/expectation) and `savvi` are Python packages for e-values and safe anytime valid inference, and [safestats](https://cran.r-project.org/package=safestats) implements safe tests in R. `ville` aims to offer one consistent interface across means, proportions, quantiles, variances and A/B tests, with the paper each estimator follows stated beside it and coverage tested by simulation.

## References

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

## Licence

BSD 3-Clause. See [LICENSE](LICENSE).

## Author

Shiv Bhaduri
