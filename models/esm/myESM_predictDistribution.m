function pred = myESM_predictDistribution(X)
    numSamples = size(X,1);
    sequenceLength = size(X,2);
    numSensors = size(X,3);

    simC.State = zeros(30);
    simC.Map = zeros(30);  

    % Setup sensors & positions
    gpINx = []; 
    i = 1;
    for col_x = 2:5:25
        for row_y = 2:5:30
            simC = ESM_addSensor(simC, [col_x, row_y]);   
            gpINx(1,i) = col_x;
            gpINx(2,i) = row_y;
            i = i+1;
        end  
    end

    %% ESN
    % Parametrize and create the Echo State Network
    Sparsity = 0.15;
    EchoLayerSize = 50; %125;
    SpectralRadius = 0.4;
    regDelay = 19;
    regNoise = 0;
    winScale = 1000;
    bandwidth = 3.5;

    alpha = [SpectralRadius, 0, winScale];
    ESNsize = [numSensors, EchoLayerSize, 0];
    ESN = ESN_Create(ESNsize, alpha, Sparsity, 0);

    %% ~~~~~~~~~~~~~~~~~~~~~~
    %% TRAIN
    %% ~~~~~~~~~~~~~~~~~~~~~~

    for idxSample = 1:numSamples
        sample = squeeze(X(idxSample, :, :));
        
        %% Train ESN
        ESN = ESN_adapt(ESN, sample', sample', regDelay, regNoise, 0);
    end 

    %% GP estimate of ESN weights
    combo = [];
    gpINy = zeros(numSensors, sequenceLength);
    gpGrid = zeros(size(simC.State,1),size(simC.State,2));
    
    tic();
    for i = 1:size(ESN.Wout,1)
        gpINy = ESN.Wout(i,:);
        gpOUTy = GPR_prediction2DNF(gpINx, gpINy - mean(gpINy), gpGrid, 0, bandwidth) + mean(gpINy); 
        combo(:,:,i) = gpOUTy .* (simC.Map+1);
    end % for i
    elapsed = toc();
    % reshape magic - replacing the ESN output weights
    ESN.WoutNoGPInterpol = ESN.Wout;
    ESN.Wout = squeeze(reshape(combo, [size(simC.State,1)*size(simC.State,2), 1, EchoLayerSize]))';


    %% Run a few ESN steps to initialize network
    for i = 1:20 
        ESN = ESN_applyStep(ESN, zeros(length(simC.Sensors),1));
    end 
    
    test_sample = squeeze(X(end,:,:));
    for i = 1:10
      [ESN, pred] = ESN_applyStep(ESN, test_sample(end,:)');
    end
    pred = reshape(pred, size(simC.State,1), size(simC.State,2));
    
    % Rotate back to original orientation
    pred = pred';
    % Crop to original size
    pred = pred(:,1:25);
    
    return;