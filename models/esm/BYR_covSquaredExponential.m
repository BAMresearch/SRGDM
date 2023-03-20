function K = BYR_covSquaredExponential(xx, x, lengthscale)

if nargin < 3
    lengthscale = 1;
end % if

if isempty(x)
    NoS = size(xx,2);
    K = zeros(NoS);
    
    for i = 1: NoS
        for j = i+1:NoS
            K(i,j) = exp(-sum((xx(:,i) - xx(:,j)).^2)/(2*lengthscale^2));
        end % for j
    end % for i

    K = K + K' + eye(NoS);
else
    NoS = size(x,2);
    NoSx = size(xx,2);
    NoD = size(x,1);    
    
    K = zeros(NoSx, NoS);
    for i = 1: NoD
        xx2 = repmat(xx(i,:)', 1, NoS);
        x2 = repmat(x(i,:), NoSx, 1);
        K = K + (xx2-x2).^2;
    end % for i 
    K = exp(-K/(2*lengthscale^2));
end % if