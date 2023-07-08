clc; clearvars; close all;
% Generate data set with mixture pdf
% p(x)=p(x|j=1)P1+p(x|j=2)P2 ; d=2;

%% Parameters 
m1=[1;1]; m2=[3; 3];
m=[m1 m2];
% S1=[0.1  -0.08; -0.08 0.2];
% S2=[0.1  0; 0 0.1];
S1=[0.2  0; 0  0.2];
S2=[0.1  0; 0  0.1];

S(:,:,1)=S1; S(:,:,2)=S2; 
%P1=0.5; P2=0.5; %Probabilidades a priori de las funciones mezcla
P1=0.85; P2=0.15;
P=[P1 P2];
N=500;


%% Generate Data Set
%   X:  lxN matrix whose columns are the produced vectors.
%   y:  N-dimensional vector whose i-th element indicates the
%       distribution generated the i-th vector.
sed=0;
[X,y]=mixt_model(m,S,P,N,sed);

%% Display data set and means
plot(X(1,:), X(2,:), 'o', m(1,1),m(2,1),'r d',m(1,2),m(2,2),'r d' ); title('Data Set - Mixture Model PDF - One class');
xlabel('x1');ylabel('x2');grid on;


