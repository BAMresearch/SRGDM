function [ESN, prediction] = ESN_applyStep(ESN, ESNinput)
% TODO UPDATE Apply one step an "Echo State Network" (ESN) 
% usage:    prediction = ESN_apply(ESN, ESNinput, timeOfInterest, vis)
% input:    ESN                     -- ESN trained with ESN_Adapt
%           ESNinput                -- input (column vector)
% ouput:    prediction              -- predicted results,
%                                      activations of the neurons in the
%                                      outputlayer
% version:  August 2016
% author:   Erik Schaffernicht
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% check input %%%
if nargin<2
    help ESN_apply;
    return;
end % if
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% static initialisation %%%
NumberOfEcho = size(ESN.EchoMatrix, 1);

W = ESN.EchoMatrix;
Wback = ESN.Wback;
Wout = ESN.Wout;
Win = ESN.Win;
outTransfer = ESN.OutputTransfer;

% Check if the ESN has an internal state - if not initialize one
if isfield(ESN, 'State')
    X = ESN.State;
else
    X = rand(NumberOfEcho, 1)*2-1;
end % fi

%prediction = zeros(size(Wout,2),timeOfInterest(2)+2);
%prediction = rand(size(Wout,2),1)*2-1;

ret = size(Wback, 2);

fromOutToDR = ~isempty(Wback);
fromInToDR = ~isempty(Win);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% prediction stage %%%

    xi = W * X;
    % outputlayer to dynamic reservoir
    if fromOutToDR
        error('Looping back not implemented in step mode yet');
        %yi = Wback * prediction(1:ret,i);
    else
        yi = zeros(NumberOfEcho, 1);
    end
    % inputlayer to dynamic reservoir
    if fromInToDR
        ui = Win * ESNinput;
    else
        ui = zeros(NumberOfEcho, 1);
    end
    ESN.State = tanh(xi+yi+ui);
    if outTransfer == 0
        prediction = Wout' * X;
    elseif outTransfer == 1
        prediction = tanh(Wout' * X);
    end % if
end % for i

% prediction = prediction(:,timeOfInterest(1)+1:timeOfInterest(2)+1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end of ESN_apply %%%

%% additional visualisation
% figure
% colormap gray
% for i = 1: size(X,2)
%     
%     imagesc(reshape(X(:,i),10,10));
%     M(i) = getframe;
% end % for i
% 
% movie2avi(M, 'activity.avi');