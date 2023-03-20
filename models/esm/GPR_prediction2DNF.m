function [meanPrediction, covMa, logP] = GPR_prediction2DNF(x,y, GPgrid, sigmaP, lengthscale)
% TO BE UPDATED Gaussian Process Regression using RBF-Kernel for 2D data
% usage:    logP = GPR_prediction1DNF(x,y, sigmaP, lengthscale)
% input:    x               -- feature matrix
%           y               -- regression target
%           sigmaP          -- assumed gaussian noise in the data
%           lengthscale     -- kernel bandwith
% output:   logP            -- logarithmic probability of the regression
% version:  august 2016
% author:   Erik Schaffernicht
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% static initializations %%%
%frac = diff(minmax(x)')/25;

xx1 = 1:1:size(GPgrid,1);
xx2 = 1:1:size(GPgrid,2);

NoE1 = length(xx1);
NoE2 = length(xx2);

[x1,x2] = meshgrid(xx1,xx2);
xx = [reshape(x1,1,NoE1*NoE2);reshape(x2,1,NoE1*NoE2)];
NoS = size(x,2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% compute kernel matrices %%%

Kxsx = BYR_covSquaredExponential(xx,x, lengthscale);
% More stable computation according to Rasmussen

L = chol(BYR_covSquaredExponential(x,[],lengthscale)+eye(NoS)*sigmaP)';

alpha = L'\(L\y');
meanPrediction = Kxsx * alpha;
v = L\Kxsx';

covPrediction = BYR_covSquaredExponential_diag(xx,[],lengthscale) - v'*v;
logP = -1/2*y*alpha-sum(log(diag(L)))- NoS/2*log(2*pi);

covMa = diag(covPrediction);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% formating output %%%
meanPrediction = reshape(meanPrediction,NoE2,NoE1)';


covMa = reshape(covMa,NoE2,NoE1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% visualization %%%
% surf(x1,x2, reshape(meanPrediction,NoE2,NoE1));
% hold on;
% plot3(x(1,:),x(2,:),y, 'dk');
% mesh(x1,x2, reshape(meanPrediction + sqrt(covMa),NoE2,NoE1));
% mesh(x1,x2, reshape(meanPrediction - sqrt(covMa),NoE2,NoE1));
% 
#fprintf('Log marginal likelihood: %f\n', logP);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end of GPR_predition1DNF %%%