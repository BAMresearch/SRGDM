function simConf = ESM_simpleIntEstimate(simConf)

simConf.StateSimpleEstimate = simConf.State .* 0;
simConf.StateEqsEstimate = simConf.State .* 0;
ESNInput = [];

for i = 1:length(simConf.Sensors)
    simConf.StateSimpleEstimate = simConf.StateSimpleEstimate + simConf.Sensors(i).Influence .* simConf.Sensors(i).SensorReadings(end);
    simConf.StateEqsEstimate = simConf.StateEqsEstimate + simConf.Sensors(i).EqsInfluence .* simConf.Sensors(i).SensorReadings(end);
    
    ESNInput = [ESNInput; simConf.Sensors(i).SensorReadings(end)];
end % for i

[simConf.ESN, pred] = ESN_applyStep(simConf.ESN, ESNInput);
simConf.StateEsmEstimate = reshape(pred, size(simConf.State,1), size(simConf.State,2));

% HACK! remove dust sinks from the error estimate
% 
% simConf.StateSimpleEstimate(2, 12) = 0;
% simConf.StateSimpleEstimate(3, 12) = 0;
% simConf.StateSimpleEstimate(2, 13) = 0;
% simConf.StateSimpleEstimate(3, 13) = 0;
% simConf.StateSimpleEstimate(2, 14) = 0;
% simConf.StateSimpleEstimate(3, 14) = 0;
% 
% simConf.StateEqsEstimate(2, 12) = 0;
% simConf.StateEqsEstimate(3, 12) = 0;
% simConf.StateEqsEstimate(2, 13) = 0;
% simConf.StateEqsEstimate(3, 13) = 0;
% simConf.StateEqsEstimate(2, 14) = 0;
% simConf.StateEqsEstimate(3, 14) = 0;
% 
% simConf.StateEsmEstimate(2, 12) = 0;
% simConf.StateEsmEstimate(3, 12) = 0;
% simConf.StateEsmEstimate(2, 13) = 0;
% simConf.StateEsmEstimate(3, 13) = 0;
% simConf.StateEsmEstimate(2, 14) = 0;
% simConf.StateEsmEstimate(3, 14) = 0;


simConf.ErrorSimpleEstimate(end+1) = sum(sum((simConf.StateSimpleEstimate - simConf.State).^2));
simConf.ErrorEqsEstimate(end+1) = sum(sum((simConf.StateEqsEstimate - simConf.State).^2));
simConf.ErrorEsmEstimate(end+1) = sum(sum((simConf.StateEsmEstimate - simConf.State).^2));