//////////////////////////////////////////////////////////////////////////////
// LST-Model Simple, Single Indicator Version ////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//
data {
  int<lower=1> N; // sample size
  int<lower=1> O; // measurement occasions
  int<lower=1> I; // indicators for latent trait
  int<lower=1> V; // variables in total (O * I)
  matrix[N, V] y_obs; // outcome matrix with missings (coded as -Inf)
  int N_mis_tot_y; // total number of missings across all indicators
  array[V] int N_mis_y; // number of missings per indicator
  array[V, max(N_mis_y)] int pos_mis_y; // position indicator for missings
  // free parameters
  int n_alpha_free;
  int n_lambda_t_free;
  int n_sigma_eps_free;
}
//
parameters {
  // alphas
  vector[V] alpha;
  // lambdas
  vector<lower=0>[V - I] lambda_t_free;
  // trait, standardized
  vector[N] trait_z;
  // trait standard deviation
  real<lower=0> sigma_trait;
  // measurement error standard deviation
  vector<lower=0>[V] sigma_eps;
  // vector to store imputed values (for missings)
  vector[N_mis_tot_y] y_impute;
  // for imputation: MVN parameters for all variables
  cholesky_factor_corr[V] L_corr;
  vector<lower=0>[V] sigma;
  vector[V] Mu;
}
//
transformed parameters {
  cholesky_factor_cov[V] L_Sigma = diag_pre_multiply(sigma, L_corr);
  // in case of missings: fill y with y_obs and empty values of y_impute 
  matrix[N, V] y = y_obs; // create new data matrix including observed and missing values
  if (N_mis_tot_y > 0) {
    int p_impute_y = 1; // initialize counter to index positions on y_impute
    for (v in 1:V) {
      y[pos_mis_y[v, 1:N_mis_y[v]], v] = segment(y_impute, p_impute_y, N_mis_y[v]);
      p_impute_y = p_impute_y + N_mis_y[v]; // update counter for next indicator
    }
  }
  matrix[N, V] full_data = y;
  // loadings for trait
  vector[V] lambda_t;
  for (i in 1:I) {
    for (o in 1:O) {
      int v = (i - 1) * O + o;
      if (o == 1) {
        lambda_t[v] = 1;
      } else {
        lambda_t[v] = lambda_t_free[v - i];
      }
    }
  }
  // trait variance
  real<lower=0> var_trait = sigma_trait^2;
  // latent trait, original scale
  vector[N] trait = trait_z * sigma_trait;
  // measurement error variance
  vector<lower=0>[V] var_eps = sigma_eps^2;
  // predicted values
  matrix[N, V] state;
  for (v in 1:V) {
    state[, v] = alpha[v] + lambda_t[v] * trait;
  }
}
//
model {
  // missing imputation with MVN
  for (n in 1:N) target += multi_normal_cholesky_lpdf(full_data[n, ] | Mu, L_Sigma);
  target += normal_lpdf(Mu | 0, 10);
  target += lognormal_lpdf(sigma | 0, 5);
  target += lkj_corr_cholesky_lpdf(L_corr | 1);
  // likelihood
  for (v in 1:V) target += normal_lpdf(y[, v] | state[, v], sigma_eps[v]);
  // trait distribution, standardized
  target += std_normal_lpdf(trait_z);
  // prior on trait sigma
  target += lognormal_lpdf(sigma_trait | 0, 1);
  // prior on rest of model parameters
  target += lognormal_lpdf(sigma_eps | 0, 1);
  target += normal_lpdf(alpha | 0, 10);
  target += normal_lpdf(lambda_t_free | 1, .2);
}
//
generated quantities {
  // state residual covariance matrix
  real<lower=0> PSI = var_trait;
  // error covariance matrix
  cov_matrix[V] THETA = diag_matrix(var_eps);
  // create loadings matrix for trait loadings
  matrix[V, O] LAMBDA = rep_matrix(0, V, O);
  for (i in 1:I) {
    for (o in 1:O) {
      int v = (i - 1) * O + o;
      LAMBDA[v, i] = lambda_t[v];
    }
  }
  // create Sigma for multivariate normal
  cov_matrix[V] SIGMA = LAMBDA * PSI * LAMBDA' + THETA;
  // likelihood vector
  vector[N] log_lik;
  // likelihood sum
  real log_lik_total;
  // multivariate normal likelihood
  for (n in 1:N) log_lik[n] = multi_normal_lpdf(y[n, ] | alpha, SIGMA);
  // sum likelihoods and store in log_lik_total
  log_lik_total = sum(log_lik);
  // generate predictions for posterior predictive checks
  matrix[N, V] y_rep;
  for (v in 1:V) y_rep[, v] = to_vector(normal_rng(state[, v], sigma_eps[v]));
  // posterior p-values for mean and sd (not in paper)
  vector<lower=0, upper=1>[V] mean_p;
  vector<lower=0, upper=1>[V] sd_p;
  for (v in 1:V) {
    mean_p[v] = mean(y_rep[, v]) > mean(y[, v]);
    sd_p[v] = sd(y_rep[, v]) > sd(y[, v]);}
}
