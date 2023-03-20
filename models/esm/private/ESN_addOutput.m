function ESN = ESN_addOutput(ESN, teacher, ESNinput, regress_Start, noise)

% Add an output an already existing "Echo State Network" (ESN)
% usage:    ESN = ESN_addOutput(ESN, teacher, ESNinput, regress_Start)
% input:    ESN                     -- ESN created with ESN_Create
%           teacher                 -- times series for teaching
%           ESNinput                -- input sequence
%           regress_Start           -- sample to start from for weight
%                                      computation (regression)
% ouput:    ESN                     -- struct containing the adapted ESN
%               .Wout                   - weights from echo to output layer
%               .AdditionalOutputs      - number of additional output units
% version:  april 2004
% author:   Erik Schaffernicht
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% check input %%%
if nargin<5
    if nargin<4
        help ESN_addOutput;
        return;
    end % if
    noise = 0;
end % if
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% check ESN %%%
if isfield(ESN,'AdditionalOutputs')
    ESN.AdditionalOutputs = ESN.AdditionalOutputs + size(teacher,1);
else
    ESN.AdditionalOutputs = size(teacher,1);
end % if
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% static initialisation %%%
SampleLength = size(teacher,2);
NumberOfEcho = size(ESN.EchoMatrix, 1);

W = ESN.EchoMatrix;
Wback = ESN.Wback;
Wout = ESN.Wout;
Win = ESN.Win;
outTransfer = ESN.OutputTransfer;

X = zeros(NumberOfEcho, SampleLength+1);
X(:,1) = rand(NumberOfEcho, 1)*2-1;

prediction = zeros(size(Wout,2),SampleLength+1);
%prediction = rand(size(Wout,2),1)*2-1;

ret = size(Wback, 2);

fromOutToDR = ~isempty(Wback);
fromInToDR = ~isempty(Win);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% sampling stage %%%
for i = 1: SampleLength
    xi = W * X(:,i);
    % outputlayer to dynamic reservoir
    if fromOutToDR
        yi = Wback * prediction(1:ret,i);
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
    X(:,i+1) = tanh(xi+yi+ui+vi);
    if outTransfer == 0
        prediction(:,i+1) = Wout' * X(:,i+1);
    elseif outTransfer == 1
        prediction(:,i+1) = tanh(Wout' * X(:,i+1));
    end % if
end % for i
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% weight computation stage %%%
M = X(:,regress_Start:end-1)';  
if outTransfer == 0
    T = teacher(:,1:size(M,1))';
elseif outTransfer == 1
    T = atanh(teacher(:,1:size(M,1))');
end % if
Wout = pinv(M) * T;
ESN.Wout = [ESN.Wout Wout];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end of ESN_addOutput %%%