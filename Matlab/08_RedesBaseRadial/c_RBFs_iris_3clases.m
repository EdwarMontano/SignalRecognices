clear all; clc; close all;

load fisheriris
x=meas(:,2:4)';
x=mapstd(x);
yd=zeros(3,150);
yd(1,1:50)=1;
yd(2,51:100)=1;
yd(3,101:end)=1;

[m,t]=max(yd);
scatter3(x(1,:),x(2,:),x(3,:),150,t,'filled'),

goal = 0.01;
spread = 1;
max_neurons = 24;
red=newrb(x,yd,goal,spread,max_neurons,1); % newrb adiciona una neurona a la vez

Pesos=red.iw{1,1};
Bias=red.b{1};

y=sim(red,x);
[m,p]=max(y);
figure(1);
hold on;
scatter3(x(1,:),x(2,:),x(3,:),50,p+3,'filled'),
sum(abs(t-p))
% pause;
plotconfusion(y,yd)

