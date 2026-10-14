# Worked examples

All examples are in [`examples/`](https://github.com/shivsbhaduri/ville/tree/main/examples/). Outputs below are unedited.

## Revenue per visitor: an unbounded, skewed metric

Revenue per visitor has a point mass at zero, a heavy right tail and no natural upper bound, so `AsymptoticABTest` is used.

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

The test stops at 7,040 of 100,000 visitors. Point estimates at data dependent stopping times are biased upwards, since stopping tends to occur when the estimate is large. The confidence sequence at the stopping time contains the true effect and is the quantity to report.

## Sequential tests and anytime valid p-values

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

The final line reports 500 betting tests under the null, each monitored at every observation up to $t = 2{,}000$: 3.2% reject at some point, against a nominal 5%.

## Latency percentiles

`QuantileCS` makes no distributional assumptions, which is appropriate for response times.

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

At $t = 200$ the upper bound for the 95th percentile is infinite: with about 10 observations above the quantile, the tail cannot be bounded without further assumptions.

## Twenty metrics, correlated, with e-BH

Twenty metrics from one experiment, four with nonzero effects. A per user shock common to all metrics makes the e-values dependent. e-BH is valid under arbitrary dependence without adjustment.

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
