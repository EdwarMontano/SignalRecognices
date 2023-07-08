clear all; clc; close all;

x=[0 0 1 1; ...
   0 1 0 1];
yd=[0 0 0 1];
plotpv(x,yd);
red=newp(x,yd);

red.iw{1,1}=[1 1];
red.b{1}=0.5;
Pesos=red.iw{1,1};
Bias=red.b{1};
plotpc(Pesos,Bias)
pause

red=train(red,x,yd);

Pesos_f=red.iw{1,1};
Bias_f=red.b{1};
figure(1);
plotpc(Pesos_f,Bias_f)
