function K = BYR_covSquaredExponential_diag(xx, x, lengthscale)
%
% Only calculate the diag(K), because this is what's used later.
%"""
if nargin < 3
    lengthscale = 1;
end % if

if isempty(x)
    NoS = size(xx,2);
    K = zeros(NoS);
    
    for i = 1: NoS
        %for j = i+1:NoS
        %    K(i,j) = exp(-sum((xx(:,i) - xx(:,j)).^2)/(2*lengthscale^2));
        %end % for j
        K(i,i) = exp(-sum((xx(:,i) - xx(:,i)).^2)/(2*lengthscale^2));
    end % for i

    K = K + K' + eye(NoS);
else
    % do nothing
    K = 0;
end % if