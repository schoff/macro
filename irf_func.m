function imp_resp = irf_func(beta, sigma, p, h, use_LR)

% beta: matrix of VAR coefficients arranged as [Phi1, Phi2, ..., Phip]
% sigma: covariance matrix of residuals
% p: number of lags in the VAR
% h: number of periods to estimate
% A: structural impact matrix derived from long-run restrictions

n = size(sigma, 1);         % number of variables

% Extract individual Phi matrices directly from beta.
Phi_lags = cell(p, 1);
for lag = 1:p
    Phi_lags{lag} = beta((lag-1)*n + (1:n), :);
end

if use_LR
    % Compute the Q matrix generalized for p lags
    Phi_sum = zeros(n);
    for lag = 1:p
        Phi_sum = Phi_sum + Phi_lags{lag};
    end
    Q = inv(eye(n) - Phi_sum) * sigma * transpose(inv(eye(n) - Phi_sum));

    % Compute Cholesky decomposition of Q (lower triangular matrix)
    chol_Q = chol(Q, 'lower');
    A = (eye(n) - Phi_sum) * chol_Q;
else
    % Standard Cholesky decomposition
    A = chol(sigma, 'lower');
end

% Initialize impulse response storage.
imp_resp = zeros(n, n, h);
imp_resp(:, :, 1) = A;  % Initial impact response

% Initialize MA coefficient matrices.
Psi = zeros(n, n, h);
Psi(:, :, 1) = eye(n);

% Compute MA coefficient matrices recursively
for t = 2:h
    Psi(:, :, t) = zeros(n);
    for lag = 1:min(p, t-1)
        Psi(:, :, t) = Psi(:, :, t) + Phi_lags{lag} * Psi(:, :, t-lag);
    end
end

% Compute the IRFs by applying the new structural impact matrix
for t = 1:h
    imp_resp(:, :, t) = Psi(:, :, t) * A;
end

end