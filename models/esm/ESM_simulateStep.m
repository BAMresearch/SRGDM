function [simConf] = ESM_simulateStep(simConf, sourceActivity)

%% Introduce new particles

% Background concentration
newBackgroundConcentration = simConf.Map;
randomMatrix = abs(randn(size(simConf.Map,1),size(simConf.Map,2))+3);
newBackgroundConcentration(newBackgroundConcentration>-1) = randomMatrix(newBackgroundConcentration>-1);
newBackgroundConcentration(newBackgroundConcentration<0) = 0;
simConf.State = simConf.State + newBackgroundConcentration;

% Source concentration
for i = 1: length(sourceActivity)
    if sourceActivity(i) > 0
        newSourceConcentration = simConf.Map;        
        for j = 1: size(simConf.Source(i).Location,1)
            newSourceConcentration(simConf.Source(i).Location(j,1),simConf.Source(i).Location(j,2)) = (randn(1)*50+200)*sourceActivity(i);
        end % for j
        simConf.State = simConf.State + newSourceConcentration;
    end % if
end % for i

%% Propagate particles
newState = zeros(size(simConf.State,1),size(simConf.State,2));

for i = 1: size(simConf.State,1)
    for j = 1: size(simConf.State,2)
        if simConf.Map(i,j)> -1
                        
            if simConf.AirflowDir(i,j) == -1
                    newState(i,j) = 0;
                    continue;
            end % if
            
            propagationAmount = simConf.State(i,j);
            propagationVector = zeros(9,1)+0.05;  
            propagationVector(simConf.AirflowDir(i,j)) = 0.6;
            
            while propagationAmount > eps
                                 
                    summedDeduction = 0;
                
                    % distribute to surrounding cells
                    % Direction 1                    
                    if simConf.Map(i-1,j) > -1
                        newState(i-1,j) = newState(i-1,j) + propagationAmount * propagationVector(1);
                        summedDeduction = summedDeduction + propagationAmount * propagationVector(1);
                    else
                        propagationVector(1) = 0;
                    end % if
                    
                    % Direction 2
                    if simConf.Map(i-1,j+1) > -1
                        newState(i-1,j+1) = newState(i-1,j+1) + propagationAmount * propagationVector(2);
                        summedDeduction = summedDeduction + propagationAmount * propagationVector(2);
                    else
                        propagationVector(2) = 0;
                    end % if
                    
                    % Direction 3
                    if simConf.Map(i,j+1) > -1
                        newState(i,j+1) = newState(i,j+1) + propagationAmount * propagationVector(3);
                        summedDeduction = summedDeduction + propagationAmount * propagationVector(3);
                    else
                        propagationVector(3) = 0;
                    end % if
                    
                    % Direction 4
                    if simConf.Map(i+1,j+1) > -1
                        newState(i+1,j+1) = newState(i+1,j+1) + propagationAmount * propagationVector(4);
                        summedDeduction = summedDeduction + propagationAmount * propagationVector(4);
                    else
                        propagationVector(4) = 0;
                    end % if
                    
                    % Direction 5
                    if simConf.Map(i+1,j) > -1
                        newState(i+1,j) = newState(i+1,j) + propagationAmount * propagationVector(5);
                        summedDeduction = summedDeduction + propagationAmount * propagationVector(5);
                    else
                        propagationVector(5) = 0;
                    end % if
                    
                    % Direction 6
                    if simConf.Map(i+1,j-1) > -1
                        newState(i+1,j-1) = newState(i+1,j-1) + propagationAmount * propagationVector(6);
                        summedDeduction = summedDeduction + propagationAmount * propagationVector(6);
                    else
                        propagationVector(6) = 0;
                    end % if
                    
                    % Direction 7
                    if simConf.Map(i,j-1) > -1
                        newState(i,j-1) = newState(i,j-1) + propagationAmount * propagationVector(7);
                        summedDeduction = summedDeduction + propagationAmount * propagationVector(7);
                    else
                        propagationVector(7) = 0;
                    end % if
                    
                    % Direction 8
                    if simConf.Map(i-1,j-1) > -1
                        newState(i-1,j-1) = newState(i-1,j-1) + propagationAmount * propagationVector(8);
                        summedDeduction = summedDeduction + propagationAmount * propagationVector(8);
                    else
                        propagationVector(8) = 0;
                    end % if
                    
                    newState(i,j) = newState(i,j) + propagationAmount * propagationVector(9);
                    summedDeduction = summedDeduction + propagationAmount * propagationVector(9);
                    
                    propagationAmount = propagationAmount - summedDeduction;
                    propagationVector = propagationVector/sum(propagationVector);
                    %fprintf('Sum prop %i %i %f \n',i,j,propagationAmount);
                
            end % while
        end % if
    end % for j
end % for i

simConf.State = newState;