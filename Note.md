# DD2434 - Machine Learning, Advanced Course

## Lecture 01 - Intro

- Linear algebra
  - What does the covariance matrix mean.
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
  - ML: Estimate $$ from trainning data 

