#include <RcppArmadillo.h>
#include <iomanip>
#include "helpers.h"

using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]

//TODO: replace inits with strcut instead
struct VIInits {
  double eta;
  arma::vec beta;
  arma::mat Sigma_beta;
  arma::vec delta;
  arma::mat Sigma_delta;
  double sig_inv;
  arma::vec mu_gamma;
  arma::vec mu_theta;
  arma::vec t;

  // Constructor to fill from an Rcpp::List
  VIInits(const Rcpp::List& inits) {
    eta = Rcpp::as<double>(inits[0]);
    beta = Rcpp::as<arma::vec>(inits[1]);
    Sigma_beta = Rcpp::as<arma::mat>(inits[2]);
    delta = Rcpp::as<arma::vec>(inits[3]);
    Sigma_delta = Rcpp::as<arma::mat>(inits[4]);
    sig_inv = Rcpp::as<double>(inits[5]);
    mu_gamma = Rcpp::as<arma::vec>(inits[6]);
    mu_theta = Rcpp::as<arma::vec>(inits[7]);
    t = Rcpp::as<arma::vec>(inits[8]);
  }
};

arma::mat q_sigma_beta(const arma::mat& X, const arma::mat& Bdiag, double mu_sig_inv) {
  arma::mat M = mu_sig_inv*X.t() * X + Bdiag;
  return safe_inverse(M);
}

arma::vec q_mu_beta(const arma::vec& y, const arma::mat& X, const arma::vec& delta, 
  const arma::mat& Sigma_beta, double mu_sig_inv){
return  mu_sig_inv*Sigma_beta * (X.t() * (y - delta));
}

arma::mat q_sigma_delta(const arma::mat& A, double mu_sig_inv, double mu_tau){
  int n = A.n_rows;
  return  safe_inverse(mu_sig_inv*(arma::eye(n,n)) + mu_tau * A);
}

arma::vec q_mu_delta(const arma::vec& y, const arma::mat& X, const arma::vec& mu_beta, 
  const arma::mat& Sigma_delta, double mu_sig_inv){
  return  mu_sig_inv*Sigma_delta * (y - X*mu_beta);
}

double q_mu_sig_inv(const arma::vec& y, const arma::mat& X, const arma::vec& mu_beta,
  const arma::vec& mu_delta, const arma::mat& Sigma_delta, const arma::mat& Sigma_beta, 
  double a_sigma, double b_sigma){
  int n = y.n_elem;
  double innerprod = 0.5*pow(arma::norm(y-mu_delta-X*mu_beta, 2),2);
  innerprod += 0.5*(arma::trace(Sigma_delta) + arma::trace(X.t()*X*Sigma_beta));
  return (n*0.5+a_sigma)/(innerprod + b_sigma); 
}

double q_mu_tau(const arma::vec& mu_delta, const arma::mat& Sigma_delta, const arma::mat& A, 
  double a_tau, double b_tau){
  int n = mu_delta.n_elem;
  double innerprod = 0.5*(arma::trace(A*Sigma_delta)+arma::dot(mu_delta, A * mu_delta));
  return (n*0.5+a_tau)/(innerprod + b_tau); 
}

double ELBO(arma::mat X, arma::mat Sigma_beta, arma::vec mu_beta, arma::vec mu_gamma,
  arma::vec mu_theta, arma::mat Sigma_theta, arma::vec mc_estimates, double eta_var, arma::vec xi, double lambda_0, double lambda_1)
{
int p = X.n_cols;
double sum_prior_beta = 0;
double sum_var_beta = 0;
double xi_i = 0;
double sum_term_gamma = 0;
double log_q_gamma = 0;
double mu_gamma_i = 0;
double mu_theta_i = 0;
double elbo = 0;

// Precompute lambda and log(lambda) values
double log_lambda_1 = log(lambda_1);
double log_lambda_0 = log(lambda_0);

for(unsigned i = 0; i < p; i++){
  mu_gamma_i = arma::as_scalar(mu_gamma.row(i));
  mu_theta_i = arma::as_scalar(mu_theta.row(i));
  xi_i = arma::as_scalar(xi.row(i));

  // p(beta)
  sum_prior_beta += log_lambda_1 * mu_gamma_i + log_lambda_0 * (1 - mu_gamma_i);
  sum_var_beta += (Sigma_beta(i,i) + pow(arma::as_scalar(mu_beta.row(i)),2)) * 
            (mu_gamma_i / lambda_1 + (1 - mu_gamma_i) / lambda_0);

  // p(gamma)
  sum_term_gamma += mu_theta_i * mu_gamma_i + log(logistic(xi_i, 0));
  sum_term_gamma -=  0.5 * mu_theta_i - 0.5 * xi_i;
  sum_term_gamma -= lambda(xi_i) * (Sigma_theta(i,i) + pow(mu_theta_i, 2) - pow(xi_i, 2));

  // q(theta)
  log_q_gamma += (mu_gamma_i * safe_log(mu_gamma_i) + (1. - mu_gamma_i) * safe_log(1. - mu_gamma_i));
}

arma::mat Sigma_rho = tri_diagonal(p, arma::as_scalar(mc_estimates.row(2)), 1);
double sum_term_theta = - 0.5 * (arma::trace(Sigma_rho * Sigma_theta) + arma::as_scalar(mu_theta.t() * Sigma_rho * mu_theta));

// log_p_beta
elbo -= 0.5 * (sum_var_beta + sum_prior_beta);

//log_p_gamma
elbo += sum_term_gamma;

// log_p_theta
elbo += sum_term_theta + 0.5 * arma::as_scalar(mc_estimates.row(0));

// log_p_eta
elbo += arma::as_scalar(mc_estimates.row(3)) - 2 * arma::as_scalar(mc_estimates.row(1)) - arma::as_scalar(mc_estimates.row(2)) * cross_term(mu_theta, Sigma_theta);

// log_q_eta (should be minus)
elbo -= log_q_gamma;

// Log-determinant terms
elbo += 0.5 * (log(eta_var) + arma::log_det(Sigma_theta).real() + arma::log_det(Sigma_beta).real());

return elbo;
}

//' LLM 
//'
//' Computes mean field variational bayes estimates of a LLM with structured spike and slab priors with p fixed and n random effect coefficients.
//'
//' @param y vector. Dependent measurements.
//' @param X Matrix. Fixed effects model matrix
//' @param A Matrix. Covariance structure of random effects = tau * A
//' @param lambda_0. Variance of spike distribution
//' @param lambda_1. Variance of slab distribution
//' @param v vector. Eigenvectors of tridiagonal pxp matrix with 0s on diagonal and 1 on the off-diagonal
//' @param inits List. Order is eta, mu_beta, Sigma_beta, mu_delta, Sigma_delta sig_inv, tau, mu_gamma, mu_theta, t
//' @param epsilon = 1e-4. ELBO convergence
//' @param max_iter = 200. Maximum number of iterations
//' @param a_sigma = 1. Hyper parameter of sigma (shape)
//' @param b_sigma = 1. Hyper parameter of sigma (scale)
//' @param a_tau = 1. Hyper parameter of tau (shape)
//' @param b_tau = 1. Hyper parameter of tau (scale)
//'
//' @return List of results
//'
//' @export
// [[Rcpp::export]]
List vi_gauss(arma::vec y, arma::mat X, arma::mat A, double lambda_0, double lambda_1, 
        arma::vec v, List inits,  double epsilon = 1e-4, int max_iter = 200, 
        double a_sigma = 1.0, double b_sigma = 1.0, double a_tau = 1.0, double b_tau = 1.0)
{
  int p = X.n_cols;
  arma::vec elbo = arma::zeros(max_iter);
  double eta = inits[0];
  arma::mat Sigma_rho = tri_diagonal(p, logistic(eta,-0.5), 1);
  arma::vec mu_beta = inits[1];
  arma::mat Sigma_beta = inits[2];
  arma::vec mu_delta = inits[3];
  arma::mat Sigma_delta = inits[4];
  double sig_inv = inits[5];
  double tau = inits[6];
  arma::vec mu_gamma = inits[7];
  arma::vec mu_theta = inits[8];
  arma::vec t = inits[9];
  arma::mat B = arma::zeros(p,p);
  arma::mat Sigma_theta = arma::zeros(p,p);
  arma::vec mc_estimates = arma::vec({0, 0, logistic(eta,-0.5), 0});

  double eta_var  = 1;
  double diff = 0;
  double temp_elbo = 0;
  bool flag = true;
  int iter = 0;

  while(flag && iter < max_iter){
    if(iter > 0 && iter % 1 == 0){
      print_iter_info(iter, elbo.row(iter-1), diff, eta);
    }

    // Fixed effects
    B = beta_mat(mu_gamma, lambda_0, lambda_1);
    Sigma_beta =  q_sigma_beta(X, B, sig_inv);
    mu_beta = q_mu_beta(y, X, mu_delta, Sigma_beta, sig_inv);
    // random effects
    Sigma_delta = q_sigma_delta(A, sig_inv, tau);
    mu_delta = q_mu_delta(y, X, mu_beta, Sigma_delta, sig_inv);
    // scales
    sig_inv = q_mu_sig_inv(y, X, mu_beta, mu_delta, Sigma_delta, Sigma_beta, a_sigma, b_sigma);
    tau = q_mu_tau(mu_delta,Sigma_delta, A, a_tau, b_tau);
    // SSL hyper params
    mu_gamma = q_gamma(mu_theta, mu_beta, Sigma_beta, lambda_0, lambda_1);
    Sigma_theta = q_sigma_theta(arma::as_scalar(mc_estimates.row(2)), p, t);
    mu_theta = q_mu_theta(mu_gamma, Sigma_theta);
    t = xi(mu_theta, Sigma_theta);
    eta = eta_hat(eta, mu_theta, Sigma_theta, v, 20, 1e-6);
    eta_var = -1/hess_q_eta(eta, mu_theta, Sigma_theta, v);
    mc_estimates = E_elbo_eta(eta, eta_var, v, 50000);

    // ELBO
    temp_elbo = ELBO(X,Sigma_beta,mu_beta,mu_gamma,mu_theta,Sigma_theta,mc_estimates, 
      eta_var,t,lambda_0,lambda_1);

    if(std::isnan(temp_elbo) || std::isinf(temp_elbo)){
      throw std::invalid_argument("ELBO is nan/inf, try different initial values");
    }

    elbo.row(iter).fill(temp_elbo);

    if(iter > 1){
      diff = arma::as_scalar(elbo.row(iter)) - arma::as_scalar(elbo.row(iter-1));
      if(iter > 5 && sqrt(pow(diff,2)) < epsilon){
        flag = false;
      }
    }
    iter += 1;
  }
  
  List L = List::create(_["elbo"] = elbo.head_rows(iter), 
    _["eta"] = eta, _["eta_var"] = eta_var, _["beta"] = mu_beta, 
    _["Sigma_beta"] = Sigma_beta, _["delta"] = mu_delta, _["Sigma_delta"] = Sigma_delta, 
    _["mu_gamma"] = mu_gamma, _["sig_inv"] = sig_inv, _["tau"] = tau,
    _["mu_theta"] = mu_theta, _["Sigma_theta"] = Sigma_theta,
    _["t"] = t);

  return L;
}