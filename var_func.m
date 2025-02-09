function [beta, u, sigma, cholesky] = var_func(Y, p)


% OLS estimation of VAR
% beta: coefficients matrix
% u: residuals matrix
% sigma: residual covariance matrix
% cholesky: cholesky decomposition of sigma

T = length(Y);
n = size(Y,2); % Number of variables in the system

% Construct lagged matrices
X = zeros(T - p, n * p);
for lag = 1:p
    X(:, (n*(lag-1)+1):(n*lag)) = Y(p+1-lag:T-lag, :); % Append lagged variables
end
y = Y(p+1:T,:); % dependent variable matrix

% OLS estimation
beta = (X' * X) \ (X' * y);
u = y - (X * beta); % residuals
sigma = (u' * u) / T; % residual covariance matrix

% Cholesky decomposition of sigma
cholesky = chol(sigma, "lower");
end