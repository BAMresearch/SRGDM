function simConf = ESM_initializeSim(scenarioName)

switch scenarioName
    case 'SimpleScenario'
        simConf.State = zeros(15);
        
        simConf.Map = zeros(15);
        simConf.Map(7:9,7:9) = -1;
        simConf.Map(1:15,1) = -1;
        simConf.Map(1:15,15) = -1;
        simConf.Map(1,1:15) = -1;
        simConf.Map(15,1:15) = -1;
        
        simConf.AirflowDir = zeros(15);
        simConf.AirflowDir(2:3,2:11) = 3;
        simConf.AirflowDir(2:3,12:14) = -1;
        simConf.AirflowDir(4,2:10) = 3;
        simConf.AirflowDir(4,11) = 2;
        simConf.AirflowDir(4:14,12:14) = 1;
        simConf.AirflowDir(5:14,2:5) = 1;
        simConf.AirflowDir(11:14,5:11) = 7;
        simConf.AirflowDir(5:10,10:11) = 1;
        simConf.AirflowDir(10,11) = 8;
        simConf.AirflowDir(5:6,6:9) = 2;
        simConf.AirflowDir(10,7:9) = 7;
        simConf.AirflowDir(7:10,6) = 8;
        
        % Adjustment
        simConf.AirflowDir(4,4) = 2;
        
        simConf.Source(1).Location = [1,1;1,2;1,3;2,1;2,2;2,3;3,1;3,2;3,3]+1;
        simConf.Source(2).Location = [12,2;12,3;12,4;13,2;13,3;13,4;14,2;14,3;14,4];
        simConf.Source(3).Location = [13,11;13,12;14,11;14,12];
    case 'NoSource'
        simConf.State = zeros(15);
        
        simConf.Map = zeros(15);
        simConf.Map(7:9,7:9) = -1;
        simConf.Map(1:15,1) = -1;
        simConf.Map(1:15,15) = -1;
        simConf.Map(1,1:15) = -1;
        simConf.Map(15,1:15) = -1;
        
        simConf.AirflowDir = zeros(15);
        
    case 'Johnson'
        
        experiment_date = '201602101345';
        %dataset_name=['C:\Users\Erik Schaffernicht\Documents\MATLAB\RAISE\DataSet_JohnsonVic\DataSet_' experiment_date];
        map_spec_name=['C:\Users\Erik Schaffernicht\Documents\MATLAB\RAISE\DataSet_JohnsonVic\maps\MapSpec_' experiment_date];
        occ_map_loc='C:\Users\Erik Schaffernicht\Documents\MATLAB\RAISE\DataSet_JohnsonVic\maps\map_johnsson3.png';
        obstacles_loc=load_occupancy_map('figure',occ_map_loc,'spec_file',map_spec_name,'plot_map',0);
        round_obstacles_loc = unique(round(obstacles_loc),'rows');
        shift_obs = round_obstacles_loc - repmat([-61 47],890,1);
        
        simConf.State = zeros(81,66);
        simConf.Map = zeros(15);
        simConf.Obstacles = abs(shift_obs);
        for i = 1:size(shift_obs,1)
            simConf.Map(abs(shift_obs(i,2)), shift_obs(i,1)) = -1;
        end % for i
    otherwise
        fprintf('Unspecified scenario\n');
        return;
end % switch