function  [ESN, X]= ESN_adapt(ESN, teacher, ESNinput, regress_Start, ...
    noise, vis)
% Adapt an "Echo State Network" (ESN)
% usage:    ESN = ESN_adapt(ESN, teacher, ESNinput, regress_Start, noise)
% input:    ESN                     -- ESN created with ESN_Create
%           teacher                 -- times series for teaching
%           ESNinput                -- input sequence
%           regress_Start           -- sample to start from for weight
%                                      computation (regression)
%           noise                   -- adds white noise to the system
%                                      DEFAULT: no noise (0)
%           vis                     -- Visualization: 1: ON (DEFAULT)
%                                                     0: OFF
% ouput:    ESN                     -- struct containing the adapted ESN
%                                      same structure like input plus:
%               .Wout                   - weights from echo to output layer
%                                         (adapted)
% version:  april 2004
% author:   Erik Schaffernicht
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% check input %%%
if nargin <6
    if nargin<5
        if nargin<4
            help ESN_adapt;
            return;
        end % if
        noise = 0;
    end % if
    vis = 1;
end % if
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% static initialisation %%%
SampleLength = size(teacher,2);
%SampleLength = size(teacher,1);
NumberOfEcho = size(ESN.EchoMatrix, 1);

W = ESN.EchoMatrix;
Wback = ESN.Wback;
Win = ESN.Win;
outTransfer = ESN.OutputTransfer;

X = zeros(NumberOfEcho, SampleLength+1);
X(:,1) = rand(NumberOfEcho, 1)*2-1;

fromOutToDR = ~isempty(Wback);
fromInToDR = ~isempty(Win);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% sampling stage %%%
for i = 1: SampleLength
    % internal dynamic reservoir
    xi = W * X(:,i);
    % outputlayer to dynamic reservoir
    if fromOutToDR
        yi = Wback * teacher(:,i);
    else
        yi = zeros(NumberOfEcho, 1);
    end
    % inputlayer to dynamic reservoir
    if fromInToDR
        ui = Win * ESNinput(:,i);
    else
        ui = zeros(NumberOfEcho, 1);
    end
    if noise
        vi = randn(1)/noise;
    else
        vi = 0;
    end % if
    % new dynamic reservoir activation
    X(:,i+1) = tanh(xi+yi+ui+vi);
end % for i
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% weight computation stage %%%
M = X(:,regress_Start:end-1)';
if outTransfer == 0
    T = teacher(:,regress_Start:end)';
elseif outTransfer == 1
    T = atanh(teacher(:,regress_Start:end)');
end % if
Wout = pinv(M) * T;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% prepare output %%%
ESN.Wout = Wout;
ESN.OutputMax = max(abs(Wout));
if ESN.OutputMax>1000
    warning('Output weights are to large! The ESN will not operate correctly.')
end %if
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% visualisation %%%

if vis == 1
    figure;
    for i = 1: min(NumberOfEcho, 20)
        subplot(5,4,i);
        plot(X(i,:));
    end
   %{ 
    figure;
    plot(M);
    hold on;
    plot(T(:,2),':'); % Plot only speed
    %}
end %if
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end of ESN_adapt %%%
%}
end