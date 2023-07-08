clear all; close all; clc;

randn('seed',0)
m1=8.75;
m2=9;
stdevi=sqrt(4);
N=1000;
x1=m1+stdevi*randn(1,N);
x2=m2+stdevi*randn(1,N);

rho=0.05;
% rho=0.001;
[h] = ttest2(x1,x2,rho);

if h==1
    msgbox(strcat('Característica selecionada, las medias son diferentes con un nivel de significancia =',num2str(rho)));
else
    msgbox(strcat('Característica rechazada, las medias son iguales con un nivel de significancia =',num2str(rho)));
end
