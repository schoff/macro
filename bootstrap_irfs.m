function [IRF_median, IRF_lower, IRF_upper] = bootstrap_irfs(beta, mu, T, nburn, y0, p, h, rep, use_LR)
% bootstrap_irfs computes bootstrap confidence intervals for IRFs.
%
%   beta: estimated VAR coefficients (from original estimation)
%   mu: residuals from the original VAR
%   T: sample size of the original data
%   nburn: number of burn-in observations for bootstrapping
%   y0: initial values (matrix with p rows, number of variables columns)
%   p: lag order of the VAR
%   h: horizon for the IRFs
%   rep: number of bootstrap replications
%   A: cholesky decomp matrix
%
% output:
%   IRF_median: median impulse responses (n x n x h)
%   IRF_lower: lower (5th percentile) impulse responses (n x n x h)
%   IRF_upper: upper (95th percentile) impulse responses (n x n x h)

n = size(y0, 2);           % number of variables
IRF_boot = zeros(n, n, h, rep);
rng(80802)

for r = 1:rep
    % Generate a bootstrap sample using quick_boot function.
    ysamp = quick_boot(beta, mu, T, nburn, y0);
    
    % Re-estimate the VAR on the bootstrap sample.
    [phi_boot, ~, sigma_boot, ~] = var_func(ysamp, p);
    
    if use_LR
        irfs_boot = irf_func(phi_boot, sigma_boot, p, h, true);
    else
        irfs_boot = irf_func(phi_boot, sigma_boot, p, h, false);
    end
    
    % For Δy (assumed to be the first variable), convert changes to levels.
    irfs_boot(1,1,:) = cumsum(irfs_boot(1,1,:), 3);
    irfs_boot(1,2,:) = cumsum(irfs_boot(1,2,:), 3);
    
    % Store the bootstrap IRFs.
    IRF_boot(:,:,:,r) = irfs_boot;
end

% Compute the median and the 5th and 95th percentiles along the replication dimension.
IRF_median = median(IRF_boot, 4);
IRF_lower  = quantile(IRF_boot, 0.05, 4);
IRF_upper  = quantile(IRF_boot, 0.95, 4);

end
