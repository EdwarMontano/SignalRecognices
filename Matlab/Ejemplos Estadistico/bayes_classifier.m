function [z]=bayes_classifier(m,S,P,X)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
%   [z]=bayes_classifier(m,S,P,X)
% Bayesian classification rule for c classes, modeled by Gaussian
% distributions 
%
% INPUT ARGUMENTS:
%   m:      lxc matrix, whose j-th column is the mean of the j-th class.
%   S:      lxlxc matrix, where S(:,:,j) corresponds to
%           the covariance matrix of the normal distribution of the j-th
%           class.
%   P:      c-dimensional vector, whose j-th component is the a priori
%           probability of the j-th class.
%   X:      lxN matrix, whose columns are the data vectors to be
%           classified.
%
% OUTPUT ARGUMENTS:
%   z:      N-dimensional vector, whose i-th element is the label
%           of the class where the i-th data vector is classified.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[l,c]=size(m);
[l,N]=size(X);

for i=1:N
    for j=1:c
        t(j)=P(j)*comp_gauss_dens_val(m(:,j),S(:,:,j),X(:,i)); %Change S by S(:,:,j)
    end
    [num,z(i)]=max(t);
end
