function simConf = ESM_esmInterpolation(simConf)


%% Parametrize and create the Echo State Network
Sparsity = 0.15;
EchoLayerSize = 125;
SpectralRadius = 0.4;
regDelay = 20;
regNoise = 0;
winScale = 1000;

alpha = [SpectralRadius, 0, winScale];
ESNsize = [length(simConf.Sensors), EchoLayerSize, 0];
ESN = ESN_Create(ESNsize, alpha, Sparsity, 0);

%% Setup robot measurements
% TODO unify with EQS in ESM_eqsInterpolation

bandwidth = 1.5;

% RobotMeasurements.numofPositions = 22;
% 
% RobotMeasurements.Pos(1).x = 9;
% RobotMeasurements.Pos(1).y = 2;
% 
% RobotMeasurements.Pos(2).x = 9;
% RobotMeasurements.Pos(2).y = 4;
% 
% RobotMeasurements.Pos(3).x = 9;
% RobotMeasurements.Pos(3).y = 6;
% 
% RobotMeasurements.Pos(4).x = 10;
% RobotMeasurements.Pos(4).y = 9;
% 
% RobotMeasurements.Pos(5).x = 12;
% RobotMeasurements.Pos(5).y = 9;
% 
% RobotMeasurements.Pos(6).x = 14;
% RobotMeasurements.Pos(6).y = 9;
% 
% 
% RobotMeasurements.Pos(7).x = 7;
% RobotMeasurements.Pos(7).y = 10;
% 
% RobotMeasurements.Pos(8).x = 7;
% RobotMeasurements.Pos(8).y = 12;
% 
% RobotMeasurements.Pos(9).x = 7;
% RobotMeasurements.Pos(9).y = 14;
% 
% RobotMeasurements.Pos(10).x = 2;
% RobotMeasurements.Pos(10).y = 7;
% 
% RobotMeasurements.Pos(11).x = 4;
% RobotMeasurements.Pos(11).y = 7;
% 
% RobotMeasurements.Pos(12).x = 6;
% RobotMeasurements.Pos(12).y = 7;
% 
% 
% % Specplace
% RobotMeasurements.Pos(13).x = 2;
% RobotMeasurements.Pos(13).y = 2;
% 
% RobotMeasurements.Pos(14).x = 11;
% RobotMeasurements.Pos(14).y = 4;
% 
% % Inner Corners
% RobotMeasurements.Pos(15).x = 6;
% RobotMeasurements.Pos(15).y = 6;
% 
% RobotMeasurements.Pos(16).x = 10;
% RobotMeasurements.Pos(16).y = 10;
% 
% RobotMeasurements.Pos(17).x = 6;
% RobotMeasurements.Pos(17).y = 10;
% 
% RobotMeasurements.Pos(18).x = 10;
% RobotMeasurements.Pos(18).y = 6;
% 
% % Middle Corners
% RobotMeasurements.Pos(19).x = 4;
% RobotMeasurements.Pos(19).y = 4;
% 
% RobotMeasurements.Pos(20).x = 12;
% RobotMeasurements.Pos(20).y = 12;
% 
% RobotMeasurements.Pos(21).x = 4;
% RobotMeasurements.Pos(21).y = 12;
% 
% RobotMeasurements.Pos(22).x = 12;
% RobotMeasurements.Pos(22).y = 4;


% Paper config

RobotMeasurements.numofPositions = 7;

RobotMeasurements.Pos(1).x = 13;
RobotMeasurements.Pos(1).y = 9;

RobotMeasurements.Pos(2).x = 8;
RobotMeasurements.Pos(2).y = 11;

RobotMeasurements.Pos(3).x = 5;
RobotMeasurements.Pos(3).y = 14;

RobotMeasurements.Pos(4).x = 2;
RobotMeasurements.Pos(4).y = 10;

RobotMeasurements.Pos(5).x = 4;
RobotMeasurements.Pos(5).y = 6;

RobotMeasurements.Pos(6).x = 9;
RobotMeasurements.Pos(6).y = 4;

RobotMeasurements.Pos(7).x = 6;
RobotMeasurements.Pos(7).y = 2;



%% Create the input for the GP
gpINx = zeros(2,RobotMeasurements.numofPositions);
for i = 1:RobotMeasurements.numofPositions
    gpINx(:,i) =  [RobotMeasurements.Pos(i).y; RobotMeasurements.Pos(i).x];
end % for i

gpINy = zeros(length(simConf.Sensors),RobotMeasurements.numofPositions);
gpGrid = zeros(size(simConf.State,1),size(simConf.State,2));

%% Data preparation

% Load training data
td =load('TrainingData.mat');
td = td.TrainingSet;

% Generate measurement matrix - X
numberOfSamples = length(td.Sensor(1).Reading);
measurements = [];

for i = 1: length(td.Sensor)
    measurements = [measurements; td.Sensor(i).Reading];
end % for i

labels = zeros(RobotMeasurements.numofPositions, numberOfSamples);
% List of positions with measurements
for i = 1: RobotMeasurements.numofPositions
    % Will have to change with the moving robot
    for j = 1: numberOfSamples
        labels(i,j) = td.State(j).Matrix(RobotMeasurements.Pos(i).y, RobotMeasurements.Pos(i).x);
    end % for j
end % for i


%% Train ESN
ESN = ESN_adapt(ESN, labels, measurements, regDelay, regNoise, 0);

%% GP estimate of ESN weights
combo = [];

for i = 1:size(ESN.Wout,1)
    % This is TOO SLOW! HANDLE multi y inside GPR_prediction2DNF
    gpINy = ESN.Wout(i,:);
    gpOUTy = GPR_prediction2DNF(gpINx, gpINy - mean(gpINy), gpGrid, 0, bandwidth) + mean(gpINy); 
    combo(:,:,i) = gpOUTy .* (simConf.Map+1);
end % for i

% reshape magic - replacing the ESN output weights
ESN.WoutNoGPInterpol = ESN.Wout;
ESN.Wout = squeeze(reshape(combo, [size(simConf.State,1)*size(simConf.State,2), 1, EchoLayerSize]))';

%% Run a few ESN steps to initialzie network
for i = 1:50 
    ESN = ESN_applyStep(ESN, zeros(length(simConf.Sensors),1));
end % for i

% reshaping back into map space - reshape(result, size(simConf.State,1), size(simConf.State,2))

% 
% % interpolate
% [xx,yy] = meshgrid(1:size(simConf.State,1),1:size(simConf.State,2));
% 
% 
% % Compute the kernel matrix for each robot measurement position
% for i = 1:RobotMeasurements.numofPositions
%     xmod = xx - RobotMeasurements.Pos(i).x;
%     ymod = yy - RobotMeasurements.Pos(i).y;
%     
%     RobotMeasurements.Pos(i).KernelMatrix = exp(-(xmod.^2 + ymod.^2)/(2*bandwidth^2));
%     combinedKernel = combinedKernel + RobotMeasurements.Pos(i).KernelMatrix;
% end % i
% 
% % NW estimate
% for i = 1:length(simConf.Sensors)
%     eqs = zeros(size(simConf.State,1),size(simConf.State,2));
%     for j = 1:RobotMeasurements.numofPositions
%         eqs =  eqs + RobotMeasurements.Pos(j).KernelMatrix .* simConf.Sensors(i).EqsInfluence(RobotMeasurements.Pos(j).y, RobotMeasurements.Pos(j).x);
%     end % for j
%     simConf.Sensors(i).EqsInfluence = eqs ./ combinedKernel;
% end % for j
% 
% if FixedInfluence > 0 
%     eqs = zeros(size(simConf.State,1),size(simConf.State,2));
%     for j = 1:RobotMeasurements.numofPositions
%         eqs =  eqs + RobotMeasurements.Pos(j).KernelMatrix .* simConf.EqsFixedInfluence(RobotMeasurements.Pos(j).y, RobotMeasurements.Pos(j).x);
%     end % for j
%     simConf.EqsFixedInfluence = eqs ./ combinedKernel;
%     simConf.EqsFixedInfluence = simConf.EqsFixedInfluence .* (simConf.Map+1);
% end % if
% 
% % mask obstacles
% for i = 1:length(simConf.Sensors)
%     simConf.Sensors(i).EqsInfluence = simConf.Sensors(i).EqsInfluence .* (simConf.Map+1);
% end % for i

simConf.ESN = ESN;
simConf.ErrorEsmEstimate = [];