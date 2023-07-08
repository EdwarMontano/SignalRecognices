clear all; close all; clc;

% [x,t]=iris_dataset;
x = csvread('../Dataset/Train_InputT450.csv');
t=zeros(3,450);
t(1,1:100)=1;
t(2,101:200)=1;
t(3,201:300)=1;
t(1,301:350)=1;
t(2,351:400)=1;
t(3,401:450)=1;

net =  patternnet(10);
net.layers{1}.transferFcn='softmax' %con esta lnea se cambia la funcin de activacin de la primera capa a ReLU
[net, tr] = train(net,x,t);
y = net(x);
perf = perform(net,t,y);
figure;
plotperform(tr);
figure;
plotconfusion(t,y);

figure; 
plot(tr.perf,'s-b','MarkerSize',6);
grid on;hold on;
plot(tr.vperf,'s-m','MarkerSize',6);
plot(tr.tperf,'s-g','MarkerSize',6)
legend('Entrenamiento','Validacin','Prueba');
ylabel('mse'); xlabel('iteracin');
title(strcat('el desempeo alcanzado fue de',num2str(perf)));

% %entrenamiento con gradiente descendiente y momento
% net = feedforwardnet(10,'traingdm'); 
% net.trainParam.lr = 0.05; %rata de aprendizaje
% net.trainParam.mc = 0.9;  %constante del momento
%entrenamiento con gradiente conjugado, hay 3 algoritmos con diferentes
%estrategias
% net = feedforwardnet(10,'traincgf'); %ver tambin 'traincgb' y 'traingcp'
