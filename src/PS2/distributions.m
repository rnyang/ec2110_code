% Generate Distributions using
% rand and randn functions

% Number of Draws
SIZE = 1000;
distribution = 't';
params = {5};

obs = drawDist(distribution, SIZE, params);

% Set Observations from Distribution

obs = x(:,1);
n = size(obs,1);

% Plot Histogram

hist(obs);

% Calculate Empirical Distribution Function %

edf = @(m) mean(obs < m);

edfvalues = (1:n) / n;

plot(sort(obs), edfvalues);

% Calculate Empirical Quantile Function %

equant = @(m) quantile(obs, m);

plot(edfvalues, sort(obs));

% Verify Jensen's Inequality

h = @(x) x .^ -1;
jensen = mean(h(x)) >= h(mean(x));

h = @(x) x .^ 2;
jensen = mean(h(x)) >= h(mean(x));

% Verify Chebychev's Inequality

M = 2;

cheb = mean(abs(x - mean(x)) >= M) <= (var(x) / (M^2));

