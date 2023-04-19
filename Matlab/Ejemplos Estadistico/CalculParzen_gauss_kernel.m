function [px]=CalculParzen_gauss_kernel(X,h)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
%   [px]=CalculParzen_gauss_kernel(X,h)
% Parzen approximation of a one-dimensional pdf using a Gaussian kernel.
%
% INPUT ARGUMENTS:
%   X:              an 1xN vector, whose i-th element corresponds to the
%                   i-th data point.
%   h:              this is the step with which the x-range is sampled and
%                   also defines the volume of the Gaussian kernel.
%   
% OUTPUT ARGUMENTS:
%   px:             a vector, each element of which contains the estimate
%                   of p(x) for each x in the range [xleftlimit,
%                   xrightlimit]. The step for x is equal to h.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[d,N]=size(X);
px=zeros(1,N);
for k=1:N
    px(k)=0;
    for i=1:N
        xi=X(:,i);
        px(k)=px(k)+exp(-(X(k)-xi)'*(X(k)-xi)/(2*h^2));
    end
    px(k)=px(k)*(1/N)*(1/(((2*pi)^(d/2))*(h^d)));
end

