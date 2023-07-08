clear all;close all; clc;

load fisheriris;
X = meas;
y = species;

c = cvpartition(y,'k',10);
opts = statset('display','iter');
fun = @(XT,yT,Xt,yt)...
      (sum(~strcmp(yt,classify(Xt,XT,yT,'quadratic'))));

[fs1,history1] = sequentialfs(fun,X,y,'cv',c,'options',opts,'direction','forward','nfeatures',3)
% [fs2,history2] = sequentialfs(fun,X,y,'cv',c,'options',opts,'direction','backward','nfeatures',3)