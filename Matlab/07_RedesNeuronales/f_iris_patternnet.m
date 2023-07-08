clear all; close all; clc;
 
[x,t]=iris_dataset;
net =  patternnet(10);
net.layers{1}.transferFcn='poslin' %con esta l�nea se cambia la funci�n de activaci�n de la primera capa a ReLU
[net, tr] = train(net,x,t);
y = net(x);
perf = perform(net,t,y);
figure;
plotconfusion(t,y);
