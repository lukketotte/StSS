#include <RcppArmadillo.h>
#include "helpers.h"
#include <iomanip>

using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]

//' Compute logistic with lower bound
// [[Rcpp::export]]
double logistic(double x, double lower){
    return 1.0 / (1.0 + std::exp(-x)) + lower;
}

arma::vec logistic(const arma::vec& x, double lower) {
    return 1.0 / (1.0 + arma::exp(-x)) + lower;
}

//' Compute logit with lower bound
// [[Rcpp::export]]
double logit(double x, double lower){
    return log((lower-x)/(lower + x));
  }
  
//' From the Jaakkola lower bound
double lambda(double x){
    return (logistic(x,0) - 0.5)/(2*x);
}
  
arma::vec lambda(const arma::vec& x) {
    return (logistic(x, 0) - 0.5) / (2 * x);
}

//' Avoiding overflow
arma::vec exp_normalize(const arma::vec& x){
    double b = x.max();
    arma::vec y = arma::exp(x - b);
    return y / arma::sum(y);
}

void print_iter_info(int iter, const arma::vec& elbo, double diff, double eta) {
    Rcout << "iter: " << iter << ", elbo: " 
          << std::scientific << std::setprecision(4) 
          << arma::as_scalar(elbo)
          << " " << (diff > 0 ? "↑" : "↓")
          << " eta: "
          << std::fixed << std::setprecision(3)
          << eta 
          << std::endl;
}

//' Computes tridiagonal matrix, rho = offdiag
// [[Rcpp::export]]
arma::mat tri_diagonal(int n, double rho, double diagVal){
    arma::mat returnMat = arma::eye(n, n) * diagVal; 
    returnMat.diag(-1).fill(rho);
    returnMat.diag(1).fill(rho);
    return returnMat;
}


  //' Computes block diagonal matrix for the Poisson case
arma::mat block_diagonal(const arma::vec& muInd, double lambda_0, double lambda_1, double sigmaInv, int n){
    int p = muInd.n_elem;
    arma::mat B = arma::eye(n + p, n + p);
    B.submat(0, 0, p - 1, p - 1) = arma::diagmat((1 - muInd) / lambda_0 + muInd / lambda_1);
    B.submat(p, p, n + p - 1, n + p - 1) = arma::diagmat(arma::vec(n).fill(sigmaInv));
    return B;
  }

arma::mat block_diagonal(const arma::vec& muInd, double lambda_0, double lambda_1, const arma::mat& SigmaInvMat, double sigmaInv){
    int p = muInd.n_elem;
    int n = SigmaInvMat.n_rows;
    arma::mat B = arma::zeros(n + p, n + p);
    B.submat(0, 0, p - 1, p - 1) = arma::diagmat((1 - muInd) / lambda_0 + muInd / lambda_1);
    B.submat(p, p, n + p - 1, n + p - 1) = sigmaInv*SigmaInvMat;
    return B;
}

//' Computes the diagonal matrix of indiciators for beta
// [[Rcpp::export]]
arma::mat beta_mat(const arma::vec& muInd, double lambda_0, double lambda_1){
    return arma::diagmat((1 - muInd) / lambda_0 + muInd / lambda_1);
}

arma::mat safe_inverse(arma::mat A){
    try {
      return arma::inv_sympd(A);  
    } catch (const std::exception& e) {
        // Check if A is symmetric
        if (!A.is_symmetric()) {
            Rcpp::Rcout << "Matrix is not symmetric, using general inversion.\n";
            A.diag() += 1e-16;  // Add small regularization
            Rcpp::Rcout << std::scientific << std::setprecision(4) <<  "Condition number = " << arma::cond(A) << "\n";
            return arma::inv(A);  
        }
  
        // If A is symmetric but still fails, regularize and try again
        Rcpp::Rcout << "Matrix is nearly singular, adding regularization.\n";
        A.diag() += 1e-16;  
        try {
            return arma::inv_sympd(A);
        } catch (...) {
            return arma::inv(A);  // Final fallback if everything fails
        }
    }
}
double safe_log(double x){
    return std::log(std::max(x, 1e-7));  // Using max to avoid log(0)
}
  
arma::vec safe_log(const arma::vec& x){
    return arma::log(arma::clamp(x, 1e-7, arma::datum::inf));  // Clamp values between 1e-7 and infinity to avoid log(0)
}


//' Compute a adjencey graph
// [[Rcpp::export]]
arma::mat compute_weight_matrix(const arma::mat& coords, double cutoff) {
    int n = coords.n_rows;
    arma::mat W(n, n, arma::fill::zeros);
  
    for (int j = 0; j < n; ++j) {
      for (int i = j; i < n; ++i) {
        arma::rowvec d = coords.row(i) - coords.row(j);
        double dist2 = arma::dot(d, d);
  
        if (dist2 < cutoff && dist2 > 0.0) {
          W(j, i) = 1.0;
          W(i, j) = 1.0;
        }
      }
    }
  
    return W;
}

arma::vec xi(arma::vec mu, arma::mat Sigma) {
    return arma::sqrt(Sigma.diag() + arma::square(mu));
}
  
double cross_term(const arma::vec& mu, const arma::mat& Sigma){
    return arma::accu(Sigma.diag(-1) + mu.head(mu.n_elem - 1) % mu.tail(mu.n_elem - 1));
}

// [[Rcpp::export]]
double hess_q_eta(double eta, arma::vec mu, arma::mat Sigma, arma::vec v){
    double dlogdet = 0;
    double num = 0;
    double denom = 0;
    double vi = 0;
    int p = arma::size(mu)[0];
    for(unsigned i = 0; i < p; i++){
        vi = arma::as_scalar(v.row(i));
        num = 2*exp(eta)*vi*(exp(2*eta)*(vi+2)+vi-2);
        denom = pow(exp(eta)+1,2)*pow(exp(eta)*(vi+2)-vi+2,2); 
        dlogdet += num/denom;
      }
    double dcross = cross_term(mu, Sigma) * (-exp(eta)*(exp(eta)-1))/pow(exp(eta)+1,3);
    return -0.5*dlogdet - dcross - 2*exp(eta)/pow(exp(eta)+1,2);
}
  
arma::vec grad_hess_q_eta(double eta, const arma::vec& mu, const arma::mat& Sigma, const arma::vec& v){
    double exp_eta = std::exp(eta);
    double logistic_eta = logistic(eta, -0.5);
    
    arma::vec term1 = 1.0 / (1.0 + logistic_eta * v);
    arma::vec dlogdet = term1 % (exp_eta * v / std::pow(exp_eta + 1.0, 2));
    
    double grad = 0.5 * arma::accu(dlogdet) - cross_term(mu, Sigma) * exp_eta / std::pow(exp_eta + 1, 2) + 1 - 2 * exp_eta / (1.0 + exp_eta);
    double hess = -0.5 * arma::accu(2 * exp_eta * v % (exp_eta * (v + 2) + v - 2) / (std::pow(exp_eta + 1, 2) * pow(exp_eta * (v + 2) - v + 2, 2))) - 
        cross_term(mu, Sigma) * (-exp_eta * (exp_eta - 1)) / std::pow(exp_eta + 1, 3) - 2 * exp_eta / std::pow(exp_eta + 1, 2);
    
    return {grad, hess};
}
  
arma::vec nr_update(double eta, arma::vec mu, arma::mat Sigma, arma::vec v) {
    int p = mu.n_elem;  
    double exp_eta = exp(eta);
    double logistic_eta = logistic(eta, -0.5);  
    double cross = cross_term(mu, Sigma);
  
    arma::vec grad_hess(2);  
    double dlogdet = 0, ddlogdet = 0;
    
    for (unsigned i = 0; i < p; i++) {
        double vi = v(i);  
        double exp_2eta = exp(2 * eta);
        
        dlogdet += 1. / (1. + logistic_eta * vi) * exp(-eta) * vi / pow(exp(-eta) + 1., 2);
        double num = 2 * exp_eta * vi * (exp_2eta * (vi + 2) + vi - 2);
        double denom = pow(exp_eta + 1, 2) * pow(exp_eta * (vi + 2) - vi + 2, 2);
        ddlogdet += num / denom;
    }
  
    grad_hess(0) = 0.5 * dlogdet - cross * exp(-eta) / pow(1. + exp(-eta), 2) + 1 - 2 * exp_eta / (1. + exp_eta);
    grad_hess(1) = -0.5 * ddlogdet - cross * (-exp_eta * (exp_eta - 1)) / pow(exp_eta + 1, 3) - 2 * exp_eta / pow(exp_eta + 1, 2);
  
    return grad_hess;
}
  
double eta_hat(double eta, arma::vec mu, arma::mat Sigma, arma::vec v, int maxIter, double eps) {
    bool flag = true;
    double oldGrad = 10000.;
    int iter = 0;
    arma::vec grads(2);  // Gradient and hessian
    
    while(flag && iter < maxIter) {
        grads = nr_update(eta, mu, Sigma, v);  // Get the gradient and hessian
        eta -= grads(0) / grads(1);  // Update eta
        
        // Check if the absolute difference in gradient is smaller than the epsilon threshold
        flag = std::abs(oldGrad - grads(0)) > eps;  
        oldGrad = grads(0);  // Store the current gradient for the next iteration
        iter++;  // Increment iteration count
    }
    
    return eta;
}

arma::vec q_gamma(const arma::vec& theta, const arma::vec& mu, const arma::mat& Sigma, double lambda_0, double lambda_1) {
    int p = theta.n_elem;
  
    arma::vec betaQuad = Sigma(arma::span(0, p-1), arma::span(0, p-1)).diag() + arma::square(mu.subvec(0, p - 1));
  
    arma::vec log_probs_0 = -0.5 * (log(lambda_1) + betaQuad / lambda_1) + theta;
    arma::vec log_probs_1 = -0.5 * (log(lambda_0) + betaQuad / lambda_0);
  
    arma::vec gammaMu = arma::exp(log_probs_0);
    gammaMu /= (gammaMu + arma::exp(log_probs_1));
    return gammaMu;
}
  
  
double mc_sigmoid(double etaHat, double Hinv, int nMC) {
    arma::vec x = rnorm(nMC, etaHat, sqrt(Hinv)); 
  
    arma::vec transformed_x = logistic(x, -0.5);  
  
    return arma::mean(transformed_x); 
}
  
arma::mat q_sigma_theta(double rho, int p, arma::vec t) {
    arma::mat sigmaRho = tri_diagonal(p, rho, 1);
    arma::vec lambda_t = 2 * lambda(t);
    arma::mat A = sigmaRho + arma::diagmat(lambda_t); 
    return safe_inverse(sigmaRho + arma::diagmat(lambda_t));
}
  
arma::vec q_mu_theta(arma::vec gamma, arma::mat SigmaTheta) {
    gamma -= 0.5; 
    return SigmaTheta * gamma; 
}

// [[Rcpp::export]]
arma::vec E_elbo_eta(double eta, double eta_var, arma::vec v, int nMC){
    arma::vec x = arma::randn(nMC, arma::distr_param(eta,sqrt(eta_var)));
    double E_log_det = 0;
    double E_log_exp = 0;
    int p = arma::size(v)[0];
    for(unsigned j = 0; j < nMC; j++){
      for(unsigned i = 0; i < p; i++){
        E_log_det += log(1 + logistic(arma::as_scalar(x.row(j)),-0.5)*arma::as_scalar(v.row(i)))/nMC;
      }
      E_log_exp += log(1+exp(arma::as_scalar(x.row(j))))/nMC;
    }
    arma::vec ret(4);
    ret.row(0).fill(E_log_det);
    ret.row(1).fill(E_log_exp);
    ret.row(2).fill(arma::mean(logistic(x, -0.5)));
    ret.row(3).fill(arma::mean(x));
    return ret;
}