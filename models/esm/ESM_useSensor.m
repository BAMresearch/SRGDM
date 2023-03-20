function simConf = ESM_useSensor(simConf)

for i = 1:length(simConf.Sensors)
    pointer = length(simConf.Sensors(i).SensorReadings);
    simConf.Sensors(i).SensorReadings(end+1) = simConf.Sensors(i).TrueReadings(pointer+1);
    simConf.Sensors(i).SensorReadingsT(end+1) = simConf.Sensors(i).TrueReadingsT(pointer+1);
end % for i