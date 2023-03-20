function simConf = ESM_simpleInterpolation(simConf)

bandwidth = 10;

[xx,yy] = meshgrid(1:size(simConf.State,2),1:size(simConf.State,1));

totalInfluence = zeros(size(simConf.State,1),size(simConf.State,2));

% Compute the influence for each sensor
for i = 1:length(simConf.Sensors)
    xmod = xx - simConf.Sensors(i).x ;
    ymod = yy - simConf.Sensors(i).y ;
    simConf.Sensors(i).Influence = exp(-sqrt((xmod.^2 + ymod.^2)/bandwidth));
    totalInfluence = totalInfluence + simConf.Sensors(i).Influence;
end % i

% normalize influence and mask obstacles
for i = 1:length(simConf.Sensors)
    simConf.Sensors(i).Influence = simConf.Sensors(i).Influence./totalInfluence;
    simConf.Sensors(i).Influence = simConf.Sensors(i).Influence .* (simConf.Map+1);
end % for i

simConf.ErrorSimpleEstimate = [];