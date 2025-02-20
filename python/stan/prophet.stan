// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

// Functions block: Defines helper functions used to compute trends with changepoints
functions {
  // Constructs a changepoint matrix A, where A[i,j] = 1 if time t[i] >= t_change[j], else 0.
  // Assumes t and t_change are sorted in ascending order.
  matrix get_changepoint_matrix(vector t, vector t_change, int T, int S) {
    matrix[T, S] A;         // Changepoint indicator matrix
    row_vector[S] a_row;    // Temporary row vector for building each row of A
    int cp_idx;             // Index to track the current changepoint

    // Initialize A as a zero matrix and a_row as a zero vector
    A = rep_matrix(0, T, S);
    a_row = rep_row_vector(0, S);
    cp_idx = 1;

    // Fill each row of A based on whether time exceeds changepoints
    for (i in 1:T) {
      while ((cp_idx <= S) && (t[i] >= t_change[cp_idx])) {
        a_row[cp_idx] = 1;  // Mark changepoint as active
        cp_idx = cp_idx + 1;
      }
      A[i] = a_row;         // Assign the row to the matrix
    }
    return A;
  }

  // Logistic trend helper functions

  // Calculates offsets (gamma) to ensure continuity in the logistic trend at changepoints
  vector logistic_gamma(real k, real m, vector delta, vector t_change, int S) {
    vector[S] gamma;        // Offsets for piecewise continuity
    vector[S + 1] k_s;      // Growth rates for each segment
    real m_pr;              // Offset from the previous segment

    // Compute growth rates by accumulating changes (delta) onto the base rate k
    k_s = append_row(k, k + cumulative_sum(delta));

    // Compute offsets to maintain continuity across segments
    m_pr = m;  // Start with the initial offset
    for (i in 1:S) {
      gamma[i] = (t_change[i] - m_pr) * (1 - k_s[i] / k_s[i + 1]);
      m_pr = m_pr + gamma[i];  // Update offset for the next segment
    }
    return gamma;
  }

  // Computes a logistic trend with changepoints, constrained by capacity
  vector logistic_trend(
    real k,              // Base growth rate
    real m,              // Trend offset
    vector delta,        // Growth rate adjustments at changepoints
    vector t,            // Time points
    vector cap,          // Carrying capacities
    matrix A,            // Changepoint matrix
    vector t_change,     // Changepoint times
    int S                // Number of changepoints
  ) {
    vector[S] gamma;     // Continuity offsets

    gamma = logistic_gamma(k, m, delta, t_change, S);
    // Return the logistic trend capped by capacity, adjusted for changepoints
    return cap .* inv_logit((k + A * delta) .* (t - (m + A * gamma)));
  }

  // Computes a piecewise linear trend with rate changes at changepoints
  vector linear_trend(
    real k,              // Base growth rate
    real m,              // Trend offset
    vector delta,        // Growth rate adjustments
    vector t,            // Time points
    matrix A,            // Changepoint matrix
    vector t_change      // Changepoint times
  ) {
    // Linear trend with slope adjustments and offset corrections
    return (k + A * delta) .* t + (m + A * (-t_change .* delta));
  }

  // Returns a constant (flat) trend equal to the offset m
  vector flat_trend(
    real m,              // Trend offset
    int T                // Number of time points
  ) {
    return rep_vector(m, T);  // Constant trend vector
  }
}

// Data block: Declares the input data required by the model
data {
  int T;                // Number of time periods
  int<lower=1> K;       // Number of regressors
  vector[T] t;          // Time points (assumed sorted in ascending order)
  vector[T] cap;        // Carrying capacities for logistic trend (must be positive when trend_indicator == 1)
  vector[T] y;          // Observed time series data
  int S;                // Number of changepoints
  vector[S] t_change;   // Times of trend changepoints (assumed sorted in ascending order)
  matrix[T,K] X;        // Regressor matrix for additive and multiplicative effects
  vector[K] sigmas;     // Prior scales for regressor coefficients
  real<lower=0> tau;    // Scale parameter for the changepoint prior
  int trend_indicator;  // Trend type: 0 = linear, 1 = logistic, 2 = flat
  vector[K] s_a;        // Indicator for additive regressors (1 = additive, 0 = otherwise)
  vector[K] s_m;        // Indicator for multiplicative regressors (1 = multiplicative, 0 = otherwise)
}

// Transformed data block: Precomputes matrices for use in the model
transformed data {
  matrix[T, S] A = get_changepoint_matrix(t, t_change, T, S);  // Changepoint matrix
  matrix[T, K] X_sa = X .* rep_matrix(s_a', T);  // Regressor matrix for additive components
  matrix[T, K] X_sm = X .* rep_matrix(s_m', T);  // Regressor matrix for multiplicative components
}

// Parameters block: Defines the parameters to be estimated
parameters {
  real k;                   // Base growth rate for the trend
  real m;                   // Trend offset (intercept)
  vector[S] delta;          // Adjustments to the growth rate at each changepoint
  real<lower=0> sigma_obs;  // Standard deviation of the observation noise
  vector[K] beta;           // Coefficients for the regressors
}

// Transformed parameters block: Computes the trend based on the specified type
transformed parameters {
  vector[T] trend;  // The trend component of the model
  // Select and compute the appropriate trend based on trend_indicator
  if (trend_indicator == 0) {
    trend = linear_trend(k, m, delta, t, A, t_change);  // Linear trend with changepoints
  } else if (trend_indicator == 1) {
    trend = logistic_trend(k, m, delta, t, cap, A, t_change, S);  // Logistic trend with capacity constraints
  } else if (trend_indicator == 2) {
    trend = flat_trend(m, T);  // Flat trend, constant across all time points
  }
}

// Model block: Specifies priors and the likelihood
model {
  // Priors for trend and noise parameters
  k ~ normal(0, 5);        // Prior for base growth rate (wide, assumes unscaled data)
  m ~ normal(0, 5);        // Prior for trend offset
  delta ~ double_exponential(0, tau);  // Laplace prior for sparse changepoint adjustments
  sigma_obs ~ normal(0, 0.5);  // Prior for noise scale (half-normal due to constraint)

  // Prior for regressor coefficients with feature-specific scales
  beta ~ normal(0, sigmas);

  // Likelihood of the observations
  // Combines the trend with multiplicative and additive regressor effects
  // Uses normal_id_glm for efficient computation of the normal likelihood
  y ~ normal_id_glm(
    X_sa,                          // Design matrix for additive regressors
    trend .* (1 + X_sm * beta),    // Trend adjusted by multiplicative regressors
    beta,                          // Coefficients for additive regressors
    sigma_obs                      // Observation noise standard deviation
  );
}