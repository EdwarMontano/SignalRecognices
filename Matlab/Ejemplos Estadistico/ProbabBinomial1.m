clc; clear all; close all;

n=50; %Tamaño de la muestra
k=3;  % cantidad de defectos en la muestra
p=0.04; % Probabilidad de observar defectuosos


Cnk = nchoosek(n,k); %número combinatorio
P=Cnk * p^k *(1-p)^(n-k)   % Probabilidad de que se presenten k defectuosos
%
y = pdf('Binomial',k,n,p)


%%

desv=1;
m=0;
x=0.0213675213675214;
f=(1/(desv*sqrt(2*pi)))* exp(-0.5*((x-m)/desv)^2)

FF=1/f




