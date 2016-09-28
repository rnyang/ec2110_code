function [ x ] = drawDist(distribution, SIZE, params)

if strcmp(distribution, 'mvnormal')

%%%
% 1a: MV Normal
%%%

mu = params{1};
sigma = params{2};

x = zeros(SIZE,2);

% Generate First Variable
x(:,1) = randn(SIZE,1) * sigma(1,1) + mu(1);

% Compute conditional distribution
mu_cond = mu(2) + (sigma(1,2)/sigma(1,1)) * (x(:,1)-mu(1));
sigma_cond = sigma(2,2) - (sigma(1,2)^2)/sigma(1,1);

% Generate Second Variable
x(:,2) = randn(SIZE,1) * sigma_cond + mu_cond;

elseif strcmp(distribution, 'chisq')

% 1b: chi-squared with k dof
% Equal to sum of k normal variables squared

k = params{1};

x = sum(randn(SIZE,k) .^ 2, 2);

elseif strcmp(distribution, 't')

% 1c: t with k dof

k = params{1};

denom = (drawDist('chisq', SIZE, {k}) ./ k) .^ -0.5;

x = randn(SIZE,1) ./ denom;

elseif strcmp(distribution, 'F')

% 1d: F with k1, k2 dof

k1 = params{1};
k2 = params{2};

top = drawDist('chisq', SIZE, {k1}) ./ k1;
bottom = drawDist('chisq', SIZE, {k2}) ./ k2;

x = top ./ bottom;

elseif strcmp(distribution,'unif')

% 1e: Sum of uniforms

k = params{1};

x = sum(rand(SIZE,k),2);

end

end

