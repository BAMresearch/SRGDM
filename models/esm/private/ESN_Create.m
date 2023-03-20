function  ESN = ESN_Create(ESNsize, alpha, sparsevalue, outTransfer)
% Create an "Echo State Network" (ESN)
% usage:    ESN = ESN_Create(ESNsize, alpha, sparsevalue, outTransfer)
% input:    ESNsize                 -- vector with number of neurons per layer
%                                      [#input,#dynamic reservoir,#output]
%           alpha                   -- scaling factors
%                                      [spectralradius of W (<1),
%                                       scaling of Wback,
%                                       scaling of Win]
%           sparsevalue             -- % of zero weights in the echo layer
%                                      (optional)
%                                      DEFAULT: full connectivity
%           outTransfer             -- transfer function of the outputlayer
%                                      neurons
%                                      0: linear (DEFAULT)
%                                      1: tangent hyperbolic
% ouput:    ESN                     -- struct containing the ESN
%               .Name                   - 'Echo State Network'
%               .EchoMatrix             - W matrix, weights of the recurrent
%                                         echo layer
%               .Wback                  - backprojection weights from output
%                                         to echo layer
%               .Win                    - weights from input to echo layer
%               .SpectralRadiusOfW      - equals alpha(1)
%               .ScaleOfWback           - equals alpha(2)
%               .ScaleOfWin             - equals alpha(3)
%               .OutputTransfer         - desired transfer function for the
%                                         outputlayer
% version:  april 2004
% author:   Erik Schaffernicht
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% check input %%%
if nargin<4
    if nargin<3
        if nargin<2
            help ESN_Create;
            return;
        end % if
        sparsevalue = 0;
    end % if
    outTransfer = 0;
end % if
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% create network %%%
ESN.Name = 'Echo State Network';
ESN.Win = (rand(ESNsize(2),ESNsize(1))*2-1)/alpha(3);
ESN.Wback = (rand(ESNsize(2),ESNsize(3))*2-1)/alpha(2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% add zero weights %%%
W0 = rand(ESNsize(2))*2-1;
WR = rand(ESNsize(2));
SparsePoints = find(WR>=1-sparsevalue);
W0(SparsePoints) = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% obtain echo state property %%%
spectralradius = max(abs(eig(W0)));
W1 = 1/spectralradius * W0;
ESN.EchoMatrix = W1 * alpha(1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% additional informations %%%
ESN.SpectralRadiusOfW = alpha(1);
ESN.ScaleOfWback = alpha(2);
ESN.ScaleOfWin = alpha(3);
ESN.OutputTransfer = outTransfer;
ESN.SparseValue = sparsevalue;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end of ESN_Create %%%