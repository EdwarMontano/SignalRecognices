clc; clearvars; close all; 
% Nmero de clases
c=2;
% Rango de los valores de la caracterstica x
x=[-3:.1:4];
Xtrain = csvread('../Dataset/Train_Input.csv');
ytrain = csvread('../Dataset/Train_Targets.csv');
Xtest = csvread('../Dataset/Test_Input.csv');
ytest = csvread('../Dataset/Test_Targets.csv');
data1=Xtrain(1:100,1);
data2=Xtrain(101:200,2);
meanData1=mean(data1);
meanData2=mean(data2);
stdData1=std(data1);
stdData2=std(data2);
% Clase 1 : gaussian pdf
u1=meanData1 ; sigm1= stdData1; %

px_w1= normpdf(data1,u1,sigm1); %pdf conditional - clase 1
Pw1=0.5;    % a priori probability
Pw1_x=Pw1*px_w1; % A posteriori Probability (with out p(x))

% Clase 2 : gaussian pdf
u2=meanData2; sigm2=stdData2;
px_w2= normpdf(data2,u2,sigm2); %pdf conditional - clase 2
Pw2=1-Pw1; %  a priori probability
Pw2_x=Pw2*px_w2;

% Matriz de prdida

L11= 0.0 ; L12= 0.6; 
L21= 0.4 ; L22= 0.0;


% L11= 0.0 ; L12= 0.4; 
% L21= 0.6 ; L22= 0.0;

% L11= 0.0 ; L12= 0.0001; 
% L21= 0.9999 ; L22= 0.0;


%Umbral x --> m�nimo error de probabilidad Bayes
a= 2*(sigm1^2-sigm2^2);
b= 4*(sigm2^2*u1-sigm1^2*u2);
k= Pw2*sigm1/(Pw1*sigm2);
k1=4*(sigm1*sigm2)^2*log(k);
c= 2*(sigm1*u2)^2-2*(sigm2*u1)^2-k1;
p=[a  b  c];
%xumbral=roots(p)

xumbral=min(abs(roots(p)))

% Umbral x --> riesgo mnimo
a= 2*(sigm1^2-sigm2^2);
b= 4*(sigm2^2*u1-sigm1^2*u2);
k= ((L21-L22)/(L12-L11)) *(Pw2*sigm1)/(Pw1*sigm2); % Se adiciona el riesgo
k1=4*(sigm1*sigm2)^2*log(k);
c= 2*(sigm1*u2)^2-2*(sigm2*u1)^2 -k1;
pr=[a  b  c];
%xriesgo=roots(pr)

xriesgo=min(abs(roots(pr)))

% Visualization
figure
plot(data1,Pw1_x,'b' ,data2,Pw2_x,'r'  ); xlabel('x')
hold on;
ix=find(x>=xumbral,1);
stem(x(ix),Pw1_x(ix), 'dg');stem(x(ix),Pw2_x(ix), 'dg'); % Bayes
ir=find(x>=xriesgo,1);
stem(x(ir),Pw1_x(ir), '*k'); stem(x(ir),Pw2_x(ir), '*k'); % Riesgo
legend('P(w1,x)','P(w2,x)','', 'Umbral Bayes', 'Umbral Riesgo')






