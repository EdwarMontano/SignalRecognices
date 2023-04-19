clc; clearvars; close all; 
% Número de clases
c=2;
% Rango de los valores de la característica x
x=[-3:.1:4];
% Clase 1 : gaussian pdf
u1=0 ; sigm1= sqrt(1/2); % 
px_w1= normpdf(x,u1,sigm1); %pdf conditional
Pw1=0.5;    % a priori probability
Pw1_x=Pw1*px_w1; % A posteeriori Probability (with out p(x))

% Clase 2 : gaussian pdf
u2=1 ; sigm2= sqrt(1/2);
px_w2= normpdf(x,u2,sigm2); %pdf conditional
Pw2=1-Pw1; %
Pw2_x=Pw2*px_w2;

% Matriz de pérdida
% L11= 0.0 ; L12= 0.6; 
% L21= 0.4 ; L22= 0.0;
% L11= 0.0 ; L12= 0.4; 
% L21= 0.6 ; L22= 0.0;

L11= 0.0 ; L12= 0.3; 
L21= 0.7 ; L22= 0.0;


%Umbral x --> mínimo error de probabilidad Bayes
a= 2*(sigm1^2-sigm2^2);
b= 4*(sigm2^2*u1-sigm1^2*u2);
k= Pw2*sigm1/(Pw1*sigm2);
k1=4*(sigm1*sigm2)^2*log(k);
c= 2*(sigm1*u2)^2-2*(sigm2*u1)^2-k1;
p=[a  b  c];
xumbral=roots(p)

% Umbral x --> riesgo mínimo
a= 2*(sigm1^2-sigm2^2);
b= 4*(sigm2^2*u1-sigm1^2*u2);
k= ((L21-L22)/(L12-L11)) *(Pw2*sigm1)/(Pw1*sigm2); % Se adiciona el riesgo
k1=4*(sigm1*sigm2)^2*log(k);
c= 2*(sigm1*u2)^2-2*(sigm2*u1)^2 -k1;
pr=[a  b  c];
xriesgo=roots(pr)

% Visualization
figure
plot(x,Pw1_x,'b' ,x,Pw2_x,'r'  ); xlabel('x')
hold on;
ix=find(x>=xumbral,1);
stem(x(ix),Pw1_x(ix), 'dg');stem(x(ix),Pw2_x(ix), 'dg'); % Bayes
ir=find(x>=xriesgo,1);
stem(x(ir),Pw1_x(ir), '*k'); stem(x(ir),Pw2_x(ir), '*k'); % Riesgo
legend('P(w1,x)','P(w2,x)','', 'Umbral Bayes', 'Umbral Riesgo')







