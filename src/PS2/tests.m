% Test Distributions

% Directory Setting
cd /Users/royang/Dropbox/Harvard/2110_Kasy/ec2110_code/src/PS2/;

SIZE = 10000;

%%% Test 1: Test Scores

% Generate test scores
mu = [0, 0];
sigma = [[1,0];[0,1]];
mathScores = drawDist('mvnormal', SIZE, {mu, sigma});

% Assign students to schools
averageScore = mean(mathScores, 2);
schoolAssignment = 1 + (averageScore > -0.4) + (averageScore > 0) + (averageScore > 0.4);

% Compute correlation
f = @(x) x(2,1);
corrFullSample = f(corr(mathScores));

corrSchool = zeros(4,1);
for i = 1:4
    corrSchool(i) = f(corr(mathScores(schoolAssignment == i,:)));
end

%%% Test 2: Parental Education

% Parental Education (1 = college, 0 = no college)
parentEduc = drawDist('unif', SIZE, {1}) > 0.5;

% Child Education
childEducChance = drawDist('unif', SIZE, {1}) > 0.25;

childEduc = (parentEduc .* childEducChance) + (1 - parentEduc) .* (1 - childEducChance);

noise = drawDist('mvnormal', SIZE, {[0,0], [[50,0];[0,50]]});

childLogEarnings = 100 * parentEduc + 150 * childEduc + noise(:,1);

% Effect of Parent Educ
f = @(x) mean(childLogEarnings(x{1})) - mean(childLogEarnings(x{2}));
parentEducDiff = f({parentEduc == 1, parentEduc == 0});
childEducDiff = f({childEduc == 1, childEduc == 0});

parentEducDiff_childcollege = f({(parentEduc == 1) .* (childEduc == 1), (parentEduc == 0) .* (childEduc == 1)})
parentEducDiff_childnocollege = f({(parentEduc == 1) .* (childEduc == 0), (parentEduc == 0) .* (childEduc == 0)})
childEducDiff_parentcollege = f({(childEduc == 1) .* (parentEduc == 1), (childEduc == 0) .* (parentEduc == 1)})
childEducDiff_parentnocollege = f({(childEduc == 1) .* (parentEduc == 0), (childEduc == 0) .* (parentEduc == 0)})