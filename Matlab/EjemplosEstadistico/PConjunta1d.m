clc; clearvars ; close all;
randn('seed',1);

% Generar Datos Clase 1 (d=1)
ND1=400; % N�mero vectores 
mu1 = 3; % Promedio
SIG1 = 1.5; % Varianza
r1 = mvnrnd(mu1,SIG1,ND1); %data set Clase 1
minr1=min(r1);maxr1=max(r1);

% Generar Datos Clase 2 (d=1)
ND2=400; % N�mero vectores
mu2 = 7;
SIG2 = 2.5;
r2 = mvnrnd(mu2,SIG2,ND2); %data set Clase 2
minr2=min(r2);maxr2=max(r2);

% Probabilidad a priori de clases 
Pw1=ND1/(ND1+ND2); Pw2=ND2/(ND1+ND2); 

%Visualizaci�n
subplot(3,2,1:2);
plot(r1, 0.1*ones(ND1),'b+',r2,-0.1*ones(ND2),'ro'); xlabel('x');
axis([min([minr1 minr2]) max([maxr1 maxr2]) -0.2 0.2]); title('Data Set (2 clases, 1d) ')

%% DENSIDADES DE PROBABILIDAD CONDICIONAL DE CLASE p(x|w)
%p(x|w1)
% n�mero bins y particionamiento eje horizontal
%Nbin=ND1/20;
Nbin=ND1/10;
delta1=(maxr1-minr1)/Nbin; %longitud bin
eje1=[minr1: delta1 : maxr1-delta1];
pxw1a=hist(r1,Nbin)/(delta1*ND1); % Se divide por delta para obtener la densidad de probabilidad
%pxw1 = medfilt1(pxw1a,3); %;%filtro mediana
pxw1 =pxw1a;
subplot(3,2,3);
plot(eje1,pxw1,'b');title('p(x|w1)');xlabel('x');

%p(x|w2)
% n�mero bins y particionamiento eje horizontal
%Nbin=ND2/20;
Nbin=ND2/10;
delta2=(maxr2-minr2)/Nbin;%longitud bin
eje2=[minr2: delta2 : maxr2-delta2];
pxw2a=hist(r2,Nbin)/(delta2*ND2);
%pxw2 = medfilt1(pxw2a,3);
pxw2=pxw2a;
subplot(3,2,4); 
plot(eje2,pxw2,'r');title('p(x|w2)');xlabel('x');


%% DENSIDAD DE PROBABILIDAD CONJUNTA p(w,x)= p(x|w) P(w)
%Pw1=0.5; Pw2=0.5; % Probabilidad a priori de clases 
%p(w1,x)
pw1x=pxw1*Pw1;
subplot(3,2,5);
plot(eje1,pw1x,'b');title('p(w1,x)= p(x|w1) P(w1)');xlabel('x');
%p(w2,x)
pw2x=pxw2*Pw2;
subplot(3,2,6);
plot(eje2,pw2x,'r');title('p(w2,x)= p(x|w2) P(w2)');xlabel('x');










