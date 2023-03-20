function prediction = ESN_apply(ESN, ESNinput, timeOfInterest, vis)
% Apply an "Echo State Network" (ESN)
% usage:    prediction = ESN_apply(ESN, ESNinput, timeOfInterest, vis)
% input:    ESN                     -- ESN trained with ESN_Adapt
%           ESNinput                -- input sequence
%           timeOfInterest          -- samples that are of interest 
%                                      [#firstsample, #lastsample]
%           vis                     -- Visualization: 1: ON (DEFAULT)
%                                                     0: OFF
% ouput:    prediction              -- predicted results,
%                                      activations of the neurons in the
%                                      outputlayer
% version:  april 2004
% author:   Erik Schaffernicht
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% check input %%%
if nargin <4
    if nargin<2
        help ESN_apply;
        return;
    end % if
    vis = 1;
end % if
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% static initialisation %%%
NumberOfEcho = size(ESN.EchoMatrix, 1);

W = ESN.EchoMatrix;
Wback = ESN.Wback;
Wout = ESN.Wout;
Win = ESN.Win;
outTransfer = ESN.OutputTransfer;

X = zeros(NumberOfEcho, timeOfInterest(2)+1);
X(:,1)= rand(NumberOfEcho, 1)*2-1;

prediction = zeros(size(Wout,2),timeOfInterest(2)+2);
prediction = rand(size(Wout,2),1)*2-1;

ret = size(Wback, 2);

fromOutToDR = ~isempty(Wback);
fromInToDR = ~isempty(Win);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% prediction stage %%%
for i = 1 : timeOfInterest(2)
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
    X(:,i+1) = tanh(xi+yi+ui);
    if outTransfer == 0
        prediction(:,i+1) = Wout' * X(:,i+1);
    elseif outTransfer == 1
        prediction(:,i+1) = tanh(Wout' * X(:,i+1));
    end % if
end % for i

prediction = prediction(:,timeOfInterest(1)+1:timeOfInterest(2)+1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% visualisation %%%
if vis == 1
    figure;
    for i = 1: min(NumberOfEcho, 20)
        subplot(5,4,i);
        plot(X(i,:));
    end
    
    t = 1:length(prediction);
    
    figure;
    j= 1;
    for i = 2:2:size(prediction,1)
        subplot(5,4,j);
        plot(t,prediction(i,:)), grid on % Only flow speed
        j=j+1;
    end % for i
end % if
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
end