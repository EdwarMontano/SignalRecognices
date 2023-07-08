clear all; close all; clc;

x = csvread('../Dataset/Train_InputT4.csv');
t=zeros(3,300);
t(1,1:100)=1;
t(2,101:200)=1;
t(3,201:300)=1;

% [x,t]=iris_dataset;
net =  patternnet(10);
net.layers{1}.transferFcn='poslin' %con esta lnea se cambia la funcin de activacin de la primera capa a ReLU
[net, tr] = train(net,x,t);
y = net(x);
perf = perform(net,t,y);
figure;
plotconfusion(t,y);
