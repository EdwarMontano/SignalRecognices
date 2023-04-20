clc; clearvars, close all;


% load('fisheriris');

% [N,d]=size(X);
% u_est=(1/N)*sum(X)'


% Sigm_est=zeros(d);
% for i=1:d
%     temp1=X(:,i)-u_est(i);
%     for j=1:d
%         Sigm_est(i,j)=sum(temp1.*(X(:,j)-u_est(j)));
%     end
% end
% Sigm_est=(1/N)*Sigm_est

% ############################################################

clc; clearvars, close all;


load('fisheriris');
X=meas
Y=species
%Generate data set X
% randn('seed',0);
% u=[2 ; -2];
% Sigm=[0.9  0.2 ; 0.2  0.3];
% N=500
% X=mvnrnd(u,Sigm, N);

% Estimate u_est
[N,d]=size(X);
u_est=(1/N)*sum(X)'


% Estimate sigm_est
Sigm_est=zeros(d);
for i=1:d
    temp1=X(:,i)-u_est(i);
    for j=1:d
        Sigm_est(i,j)=sum(temp1.*(X(:,j)-u_est(j)));
    end
end
Sigm_est=(1/N)*Sigm_est

