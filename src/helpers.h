#ifndef HELPER_H
#define HELPER_H

#include <RcppArmadillo.h>

double logistic(double x, double lower);
arma::vec logistic(const arma::vec& x, double lower);
double logit(double x, double lower);
double lambda(double x);
arma::vec lambda(const arma::vec& x);
arma::vec exp_normalize(const arma::vec& x);
arma::mat tri_diagonal(int n, double rho, double diagVal);
arma::mat beta_mat(const arma::vec& muInd, double lambda_0, double lambda_1);
arma::mat block_diagonal(const arma::vec& muInd, double lambda_0, double lambda_1, double sigmaInv, int n);
arma::mat block_diagonal(const arma::vec& muInd, double lambda_0, double lambda_1, const arma::mat& SigmaInvMat, double sigmaInv);
arma::mat safe_inverse(arma::mat A);
arma::vec xi(arma::vec mu, arma::mat Sigma);
double cross_term(const arma::vec& mu, const arma::mat& Sigma);
double hess_q_eta(double eta, arma::vec mu, arma::mat Sigma, arma::vec v);
arma::vec grad_hess_q_eta(double eta, const arma::vec& mu, const arma::mat& Sigma, const arma::vec& v);
arma::vec nr_update(double eta, arma::vec mu, arma::mat Sigma, arma::vec v);
double eta_hat(double eta, arma::vec mu, arma::mat Sigma, arma::vec v, int maxIter, double eps);
arma::vec q_gamma(const arma::vec& theta, const arma::vec& mu, const arma::mat& Sigma, double lambda_0, double lambda_1);
double mc_sigmoid(double etaHat, double Hinv, int nMC);
arma::mat q_sigma_theta(double rho, int p, arma::vec t);
arma::vec q_mu_theta(arma::vec gamma, arma::mat SigmaTheta);
double safe_log(double x);
arma::vec safe_log(const arma::vec& x);
arma::vec E_elbo_eta(double eta, double eta_var, arma::vec v, int nMC);
void print_iter_info(int iter, const arma::vec& elbo, double diff, double eta);
#endif