
using Distributions
using Plots
#using PyPlot

nDraws = 10000;

# Create Beta Random Variable
p = Beta(1,1);

# Generate probabilities p_i
p_i = rand(p, nDraws);

# Generate Uniform [0,1] to generate Bernoulli
# (easier than generating 10000 Bernoulli variables with diff probs)
u = Uniform(0,1);
u_emp = rand(u, (nDraws,3));

# Generate Bernoulli variable X
X = round(Int64, p_i .> u_emp);

# Generate n by summing across rows i
n = sum(X, 2);

for i = [0,1,2,3]
    # For n_i
    
    title = "Histogram of p_i|n_i for n_i=$(i)" 
    fig = plot(reuse=false, title=title, size=(400,100), legend=false)
    
    # Pull out p_i | n_i = $(i)
    data = p_i[vec(n .== i)];

    # Plot histogram
    histogram!(data);
    
    # Display
    display(fig);
end

# Define Quadratic Loss Function
quadLoss(a,mu)=(a-mu)^2;

# Estimates Risk Function using simulations

# decisionFn: Array{Float64} -> Number
# lossFn: Float64 -> Float64 -> Float64
# DGP: Distribution
# parameter: Float64
# nEstimates: Int64

function estimateRisk(decisionFn, lossFn, DGP, parameter, nEstimates)
    x = rand(DGP, nEstimates);
    f = u -> lossFn(decisionFn(u), parameter);
    return mean(map(f, x));
end

# Construct functions to generate decision functions
delta1_maker(alphabeta) = x -> alphabeta[1] + alphabeta[2] * x;
delta2_maker(lambda) = x -> (x < -lambda) * (x + lambda) + 
                       (x > lambda) * (x - lambda);
delta3_maker(kappa) = x -> (abs(x) > kappa) * x;

# Generate Grid for Risk Function
gridMin = -10; gridMax = 10; gridStep = 0.5;
grid = gridMin:gridStep:gridMax;

# Set Parameters
lossFn = quadLoss;
nEstimates = 10000;

variance_grid = [0.5, 1, 2];
delta_fns = [delta1_maker([0, .5]), 
             delta1_maker([0, 1]), 
            delta1_maker([.5, .5]), 
            delta1_maker([.5, 1]),
            delta2_maker(.4),
            delta2_maker(1),
            delta2_maker(2.5),
            delta3_maker(.4),
            delta3_maker(1),
            delta3_maker(2.5)];

for (d, dFn) in enumerate(delta_fns)
    for variance in variance_grid
        # Compute Risk value for grid

        f = m -> estimateRisk(dFn, lossFn, Normal(m, variance), 
                              m, nEstimates);
        Rgrid = map(f, grid)

        # Plot Risk Function
        fig = plot(grid, Rgrid, size=(400,100), label="R - Delta Fn $d, Variance $variance");
        display(fig)
    end
end

nObs = 1;

for nObs = [4,16,64,256]

    mu = zeros(2,1);
    lambda = 3.;

    # Generate Test Data
    X = Array{Float64}(nObs,2);

    # Fill first column with constant 1
    X[:,1] = 1;

    # Fill second column with uniform draws
    X[:,2] = rand(Uniform(0,3), nObs);

    beta = [1, 0.5];

    # generate errors
    e = rand(Normal(0,1), nObs);

    # generate Y
    Y = X * beta + e;

    # Compute posterior mean and variance of beta
    meanPost = (X'*X+lambda^-2*eye(2))^-1 * (lambda^-2 * mu + X' * Y);
    varPost = (X' * X + lambda^-2 * eye(2))^-1;

    chisqThreshold = quantile(Chisq(2), .1);

    n = 100;
    x = linspace(-2, 2, n);
    y = linspace(-2, 2,n);

    xgrid = repmat(x',n,1);
    ygrid = repmat(y,1,n);

    z = zeros(n,n);

    # Test against Chi-sq distribution
    for i in 1:n
        for j in 1:n
            u = (meanPost - [x[i];y[j]])
            z[i,j] = [u' * (varPost^-1) * u][1] < chisqThreshold
        end
    end

    fig = figure(figsize=(10,8))
    ax = fig[:add_subplot](1,1,1, projection = "3d")
    cp = ax[:plot_surface](xgrid, ygrid, z, 
                    rstride=2, edgecolors="k", 
                    cstride=2, alpha=0.5, linewidth=0.1)
    PyPlot.xlabel("beta_2")
    PyPlot.ylabel("beta_1")
    PyPlot.title("Posterior Credible Set")
    display(fig)

end

x = Uniform(0,4);
y = Chisq(3);
z = TDist(4);

function getEmpSampleAverage(DGP, expected, n, k)
    data = rand(DGP, n, k);
    xbar = mean(data, 1);
    xtilde = sqrt(n) * (xbar - expected);
    return (xbar, xtilde)
end

dist_grid = [(x,2,"X"), (y,3,"Y"), (z,0,"Z")];
n_grid = [4, 16, 64, 256, 1024];

for n in n_grid
    for (dist, param, distname) in  dist_grid
        (xb, xt) = getEmpSampleAverage(dist, param, 4, 10000);

        title = "Sample Average $distname n=$n" 
        fig = plot(reuse=false, title=title, size=(400,100), legend=false)
        histogram!(collect(xb));
        display(fig)

        title = "Centered Sample Average $distname n=$n"
        fig = plot(reuse=false, title=title, size=(400,100), legend=false)
        histogram!(collect(xt));
        display(fig)
    end
end

mu_grid = [0, 1];
n_grid = [4, 16, 64, 256];



mu = 0;
n = 200;

for mu in mu_grid
    for n in n_grid

        k = 10000;

        X = rand(Normal(mu, 20/n), k);
        Y = sqrt(n) * (cos(X) - cos(mu));

        title = "Y for mu=$mu and n=$n"
        fig = plot(reuse=false, title=title, size=(400,100), legend=false)
        histogram!(Y);
        display(fig)
        
    end
end
