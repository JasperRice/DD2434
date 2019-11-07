# DD2434 - Machine Learning, Advanced Course

## Lecture 01 - Intro

- Linear algebra
  - What does the covariance matrix mean?
- Product rule:

$$
p(x,y) = p(y)p(x \vert y)
$$

- Exclusive & Exhaustive:

$$
\sum_{i}p(A_{i}) = 1 \\
p(B) = \sum_{i}p(B, A_{i}) = \sum_{i}p(A_{i})p(B \vert A_{i})
$$

- Bayesian:

$$
p(M_{i} \vert X) = \frac{p(X, M_{i})}{P(X)} = \frac{p(X \vert M_{i})p(M_{i})}{\sum_{j}p(X \vert M_{j})p(M_{j})}
$$

- Maximum Likelihood (ML) and Posterior Predictive
  - ML: Estimate $\theta_{i}$ from training data 
  - ...



## Lecture 02 - Fundamentals of the Probabilistic Approach

> Bishop 1-2, in particular 2.3 and 2.4.

### Probabilistic Approach

| Bayesian                                                     | Frequentist                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Probability is a measure of belief.                          | The ratio of outcomes in repeated trials.                    |
| The data is fixed, models have probabilities.                | There is a true model and the data is a random realization.  |
| There does not have to be an experiment for declaring probability. | Parameters can only be deduced from data (likely outcome of experiment). |
| Can incorporate prior knowledge, probabilities can be updated. | Each repeated experiment starts from ignorance.              |
| Estimators are good for available data.                      | Estimators are averaged across many trials.                  |
| Probability of a hypothesis given the data (posterior distribution). | Probability of the data given hypothesis (likelihood, sampling dist.). |
| All variables/parameters have distribution.                  | Parameters are fixed unknowns that can be point estimated from repeated trials. |

#### Learning and inference

- Expectation:

$$
\mathrm{E}[x] = \int xp (x) dx
$$

- Maximum Likelihood Estimation (MLE): *log-likelihood* is often used to convert a $\prod$ to a $\sum$.

$$
p(\mathbf{t}\vert \mathbf{x}, \mathbf{w}, \beta) = \prod_{i}^{N} \mathcal{N}(t_{n} \vert y(x_{n}, \mathbf{w}), \beta^{-1})
$$

- Maximize a Posterior (MAP):

  Maximization of $\ln p(\mathbf{w}\vert \mathbf{x}, \mathbf{t}, \beta)$ would requires:
$$
\mathbf{w}_{\mathrm{MAP}} = \arg\min_{\mathbf{w}}\{\frac{\beta}{2}\sum_{n=1}^{N}(y(x_{n}, \mathbf{w})-t_{n})^{2} + \frac{\alpha}{2}\mathbf{w}^{T}\mathbf{w}\}
$$



#### The Gaussian distribution

$$
\mathcal{N}(\mathbf{x} \vert \mathbf{\mu}, \mathbf{\sigma}) = \frac{1}{(2\pi)^{\mathcal{D}/2}}\frac{1}{|\sigma|^{1/2}}\exp\left[ -\frac{1}{2}(\mathbf{x}-\mathbf{\mu})^{T}\sigma^{-1}(\mathbf{x}-\mathbf{\mu}) \right]
$$






## Lecture 03 - Linear Regression & Model Selection

> Bishop 3, in particular 3.1, 3.3 and 3.4.

Probabilistic Linear Regression: $y(x, \mathbf{w}) = w_{0} + w_{1} x$, where $\mathbf{w} \sim \mathcal{N}(\mathbf{\mu}, \sigma)$.



- Likelihood: probability of **data** given model parameters.

- Prior: probability of **model**.

- Posterior: probability of **model** parameters given data.



$$
p(\mathbf{w} | \mathcal{D}, \mathcal{M}_{i}) = \frac{p(\mathcal{D}|\mathbf{w}, \mathcal{M}_{i}) p(\mathbf{w}|\mathcal{M}_{i})}{p(\mathcal{D}|\mathcal{M}_{i})}
$$
Marginal likelihood:
$$
p(\mathcal{D}|\mathcal{M}_{i}) = \int p(\mathcal{D}|\mathbf{w}, \mathcal{M}_{i}) p(\mathbf{w}|\mathcal{M}_{i}) d\mathbf{w}
$$
Then, the posterior model evidence would be:
$$
p(\mathcal{M}_{i}|\mathcal{D}) \propto p(\mathcal{D}|\mathcal{M}_{i}) p(\mathcal{M}_{i})
$$
Based on the model evidence, we have model mixture of $K$ models (or simply comparison between models):
$$
p(t|\mathcal{x}, \mathcal{D}) = \sum_{i=1}^{K} p(t|\mathbf{x}, \mathcal{M}_{i}, \mathcal{D}) p(\mathcal{M}_{i}|\mathcal{D})
$$




## Lecture 04 & 05 - Kernels & Introduction to Gaussian Processes

> Bishop 6.1, 6.2, 6.4.

### Dual linear regression

$$
p(\mathbf{w}|\mathbf{t}, \mathbf{X}) \propto p(\mathbf{t}|\mathbf{w}, \mathbf{X}) p(\mathbf{w}|\mathbf{X})
$$





### Dual representations



### Kernel functions

akfhakfhaf