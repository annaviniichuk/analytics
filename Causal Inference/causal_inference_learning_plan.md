## Cusal Inference with Python

This plan focuses on learning core causal inference tools and implementing each of them in Python. Follow the topics in order; each section lists key concepts, Python tools, and practice ideas, with links to your existing files where relevant.

---

### **1. Two-sided t-test + Simple OLS (continuous / ratio metrics)**

- **Goal**: Learn how to compare means between two groups and model a continuous outcome using linear regression.
- **Core concepts**:
  - Difference in means, sampling distributions, standard error
  - Null vs alternative hypotheses, p-values, confidence intervals
  - Assumptions: independence, normality (CLT), homoskedasticity
  - Simple OLS: Y = beta_0 + beta_1*Treatment + error
- **Python libraries**:
  - `numpy`, `pandas`
  - `scipy.stats` (`ttest_ind`, normal approximation)
  - `statsmodels.api` / `statsmodels.formula.api` for OLS (`ols`)

**Learning steps**:

1. Simulate A/B test style data with a continuous metric (e.g., revenue, session length).
2. Run a two-sided independent t-test to compare treatment vs control.
3. Fit a simple OLS with a treatment dummy and interpret \beta_1 as the average treatment effect.
4. Check assumptions: plot residuals, check approximate normality and equal variance.

- **Practice ideas**:
  - Vary sample size and effect size; see how p-values and confidence intervals change.
  - Add one covariate (e.g., user segment) to move from simple to multiple OLS.
- **Relevant files (to create / extend)**:
  - [Classic AB test](AB_test_continious_t_test.ipynb)

---

### **2. Two-sided z-test + Logistic Regression (proportions / binary outcomes)**

- **Goal**: Learn how to compare proportions between two groups and model binary outcomes with logistic regression.
- **Core concepts**:
  - Bernoulli / binomial outcomes (e.g., conversion, click)
  - Two-sample z-test for proportions (large-sample normal approximation)
  - Logistic regression: log-odds, odds ratios, marginal effects
  - Interpreting coefficients as log-odds changes due to treatment
- **Python libraries**:
  - `numpy`, `pandas`
  - Manual z-test implementation (or `statsmodels.stats.proportion` helpers)
  - `statsmodels.api` / `statsmodels.formula.api` for logistic regression (`Logit` or `glm` with binomial family)
- **Learning steps**:
  1. Simulate binary outcome data (e.g., sign-up yes/no) for treatment and control.
  2. Implement a two-sided z-test for difference in proportions.
  3. Fit a logistic regression with treatment as predictor; interpret odds ratios and predicted probabilities.
  4. Compare z-test and logistic regression conclusions.
- **Practice ideas**:
  - Add covariates and see how the estimated treatment effect changes.
  - Plot predicted probabilities across covariate values.
- **Relevant files (to create / extend)**:
  - Add a script/notebook, e.g. `z_test_and_logistic_regression.py` in `Statistical Inferences/Causal Inference/`.

---

### **3. Interrupted Time Series (ITS)**

- **Goal**: Learn how to estimate the causal impact of a single intervention in a time series (before/after within one unit or a few units).
- **Core concepts**:
  - Time series structure: trend, seasonality, noise
  - Intervention indicator and level change vs slope (trend) change
  - ITS regression: Y_t = \beta_0 + \beta_1 \cdot \text{time}_t + \beta_2 \cdot \text{post}_t + \beta_3 \cdot (\text{time}_t \times \text{post}_t) + \epsilon_t
  - Autocorrelation and why naive standard errors can be optimistic
- **Python libraries**:
  - `pandas` (datetime index, resampling)
  - `matplotlib`, `seaborn` for visualization
  - `statsmodels` for regression (and optionally time-series diagnostic tools)
- **Learning steps**:
  1. Start from a time series (real or simulated) with a clear intervention date.
  2. Build features: time index, `post` indicator, interaction `time * post`.
  3. Fit an ITS regression and interpret level and slope changes after intervention.
  4. Visualize fitted values vs actuals; check residual autocorrelation (ACF/PACF).
- **Practice ideas**:
  - Use your `NY Yellow Taxi` time series to design an ITS-style analysis around a chosen policy/date.
  - Try both a simple ITS regression and a more explicit time-series model (e.g., ARIMA with intervention dummy).
- **Relevant files (existing)**:
  - [NY Yellow Taxi time series.ipynb](../Timeseries/NY Yellow Taxi time series.ipynb)
  - [NY Yellow Taxi time series.py](../../pybasics/NY Yellow Taxi time series.py)

---

### **4. Difference-in-Differences (DiD)**

- **Goal**: Learn how to estimate treatment effects using treated vs control groups observed before and after an intervention.
- **Core concepts**:
  - Panel / repeated measures data (unit × time)
  - Treatment indicator, post indicator, and interaction (`treated * post`)
  - Parallel trends assumption and how to partially assess it
  - Placebo tests and robustness checks
  - Basic DiD regression: Y_{it} = \alpha + \beta_1 \cdot \text{treated}_i + \beta_2 \cdot \text{post}_t + \beta_3 \cdot (\text{treated}_i \times \text{post}*t) + \epsilon*{it}
- **Python libraries**:
  - `pandas`, `numpy`
  - `statsmodels.formula.api` for OLS DiD regression
  - Optionally `linearmodels` for more advanced panel DiD
- **Learning steps**:
  1. Work through your simulated DiD example (dog toys vs cat toys, marketing intervention).
  2. Construct `treated`, `post`, and interaction `treated * post`.
  3. Estimate the DiD model, interpret the interaction coefficient as the average treatment effect.
  4. Test parallel trends using pre-period data (interaction of treatment with time trend).
  5. Run placebo tests with fake intervention dates to see if large “effects” appear where none should.
- **Practice ideas**:
  - Vary the true effect in the simulation and see how well the model recovers it.
  - Add covariates and/or unit fixed effects and compare results.
- **Relevant files (existing)**:
  - [Diff-in-diff](Diff_in_diff.ipynb) - DiD analysis with parallel trends test and placebo tests

---

### **5. CUPED (Variance Reduction for Experiments)**

- **Goal**: Learn how to reduce variance in A/B tests using pre-experiment covariates (pre-period metrics).
- **Core concepts**:
  - Pre-period vs experiment-period metric (e.g., pre-period revenue vs experiment revenue)
  - Covariance between pre-period and experiment-period metrics
  - CUPED adjustment: Y^{\text{adj}} = Y - \theta (X - \mathbb{E}[X]), where X is a pre-period covariate
  - Effect of variance reduction on detectable effect size and power
- **Python libraries**:
  - `numpy`, `pandas`
  - `statsmodels` for estimating \theta via regression
- **Learning steps**:
  1. Simulate an A/B test with a pre-period metric X and experiment metric Y.
  2. Estimate \theta by regressing Y on X (control or both groups, depending on design).
  3. Construct CUPED-adjusted outcome Y^{\text{adj}}.
  4. Re-run your t-test / OLS on the adjusted outcome and compare standard errors and p-values.
- **Practice ideas**:
  - Vary the correlation between pre-period X and outcome Y to see how much variance reduction you get.
  - Compare sample size requirements with and without CUPED for the same detectable effect.
- **Relevant files (to create / extend)**:
  - Extend the t-test/OLS example (e.g., `t_test_and_ols_example.py`) to include a pre-period metric and CUPED adjustment.

---

### **6. Event Study**

- **Goal**: Learn how to generalize DiD into an event study that traces dynamic treatment effects over time (before and after treatment).
- **Core concepts**:
  - Relative time indicators (e.g., k periods before/after treatment)
  - Event-time dummies and omitting a reference period
  - Visualizing dynamic treatment effects and checking pre-trends graphically
  - Connection to DiD (the DiD interaction is a special case of an event-study coefficient)
- **Python libraries**:
  - `pandas`, `numpy`
  - `statsmodels.formula.api` for regression with multiple time-relative dummies
- **Learning steps**:
  1. Start from your DiD simulation (dog toys vs cat toys) and construct event-time dummies (e.g., k = -3, -2, -1, 0, 1, 2,\dots).
  2. Run an event-study regression with these dummies (excluding one pre-period as the baseline).
  3. Plot the estimated coefficients and confidence intervals over event time.
  4. Interpret pre-period coefficients (parallel trends) and post-period dynamics (build-up, decay, etc.).
- **Relevant files (existing)**:
  - [Event Study](Event_study.ipynb) - Complete event study implementation with dynamic treatment effects

---

### **7. Interaction Models**

- **Goal**: Learn how to model heterogeneous treatment effects and effect modification using interaction terms in regression.
- **Core concepts**:
  - Interaction between treatment and covariates (e.g., segment, geography, device)
  - Interpretation of interaction coefficients in OLS and logistic regression
  - Subgroup effects vs fully interacted models
  - Connection to DiD (treatment × time) and event-study (treatment × event-time)
- **Python libraries**:
  - `statsmodels.formula.api` for specifying interactions using `*` and `:`
  - `pandas` for constructing categorical and numeric covariates
- **Learning steps**:
  1. Take your existing OLS or logistic regression A/B examples and add one covariate (e.g., high/low activity users).
  2. Fit models with and without treatment × covariate interaction and compare:
    - Example formula: `metric ~ treatment * segment` or `conversion ~ treatment * device`.
  3. Interpret the interaction term as a difference in treatment effect between subgroups.
  4. Compute and plot marginal effects by subgroup.
- **Practice ideas**:
  - Use your DiD or ITS setups and introduce segment-level interactions to see if some units respond more strongly than others.
  - Explore continuous moderators (e.g., baseline metric) and discretized versions (e.g., quantiles).
- **Relevant files (to create / extend)**:
  - Extend:
    - Your t-test/OLS and logistic regression examples.
    - Your DiD and ITS notebooks/scripts:
      - [DID_format.ipynb](DID_format.ipynb) - Add interaction terms (e.g., `treated * segment`)
      - [DID_example.py](DID_example.py) - Add heterogeneous effects by subgroup
      - [NY Yellow Taxi time series.ipynb](../Timeseries/NY Yellow Taxi time series.ipynb) - Add interaction models

---

### **How to Use This Plan**

- **Order**: Start with t-tests/OLS and z-tests/logistic regression, then move to ITS and finally DiD.
- **Workflow**: For each approach, (1) review concepts, (2) implement simulation in Python, (3) analyze results and assumptions, and (4) adapt the pattern to a real or semi-real dataset.
- **Extension**: After these, you can continue into propensity score methods, synthetic control, and advanced panel DiD as natural next steps.

