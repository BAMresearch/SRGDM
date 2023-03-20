function simConf = ESM_addSensor(simConf, sPosition)

if isfield(simConf, 'Sensors')
    i = length(simConf.Sensors);    
else
    i = 0;
end %if

simConf.Sensors(i+1).x = sPosition(2);
simConf.Sensors(i+1).y = sPosition(1);

simConf.Sensors(i+1).SensorReadingsT = [];
simConf.Sensors(i+1).SensorReadings = [];
simConf.Sensors(i+1).TrueReadings = [];

%% Parameters
simConf.Sensors(i+1).Offset = 75;
simConf.Sensors(i+1).Cutoff = 800;
simConf.Sensors(i+1).Noise = 30;
simConf.Sensors(i+1).Scaling = 0.9;
