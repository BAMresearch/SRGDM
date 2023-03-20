function simConf = ESM_interpolateSensors(simConf)

simConf.StateSimpleEstimate = simConf.State .* 0;
simConf.StateSimpleEstimateT = simConf.State .* 0;

for i = 1:length(simConf.Sensors)
    simConf.StateSimpleEstimate = simConf.StateSimpleEstimate + simConf.Sensors(i).Influence .* simConf.Sensors(i).SensorReadings(end);
    simConf.StateSimpleEstimateT = simConf.StateSimpleEstimateT + simConf.Sensors(i).Influence .* simConf.Sensors(i).SensorReadingsT(end);
end % for i