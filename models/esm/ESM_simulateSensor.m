function simConf = ESM_simulateSensor(simConf)

for i = 1:length(simConf.Sensors)
    x = simConf.Sensors(i).x;
    y = simConf.Sensors(i).y;
    simConf.Sensors(i).SensorReadings(end+1) = min((simConf.State(y,x)+ simConf.Sensors(i).Offset + randn(1)*simConf.Sensors(i).Noise)*simConf.Sensors(i).Scaling, simConf.Sensors(i).Cutoff);
    
    %simConf.Sensors(i).SensorReadings(end+1) = simConf.State(y,x);
    simConf.Sensors(i).TrueReadings(end+1) = simConf.State(y,x);
end % for i