clear all, clc; close all;

load fisheriris
% x=meas(1:100,2:4)'; %dos primeras clase
x=meas(51:150,2:4)'; %dos últimas clases
x=mapstd(x);
%mean(x1'),std(x1),
% yd=zeros(2,100);
% yd(1,1:50)=1;
% yd(2,51:100)=1;
% n=2;
yd=zeros(1,100);
n =1;
yd(51:100)=1;
plotpv(x,yd);

PS=[min(x(1,:)) max(x(1,:)); min(x(2,:)) max(x(2,:)); min(x(3,:)) max(x(3,:))];
red=newp(PS,n);
red.trainParam.epochs = 100;
red.trainParam.goal = 0.00009;

red=train(red,x,yd);
Pesos=red.iw{1,1};
Bias=red.b{1};
figure(1);hold on;
plotpc(Pesos,Bias)

y=sim(red,x);
error=abs(y-yd);
figure;
plotconfusion(y,yd)