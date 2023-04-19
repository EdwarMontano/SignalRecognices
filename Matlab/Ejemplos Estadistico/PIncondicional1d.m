clc; clearvars ; close all;
% Generar Datos Clase 1 (d=1)
ND1=400; % Número vectores 
mu1 = 3;  %Promedio
SIG1 = 1.5; % Varianza
r1 = mvnrnd(mu1,SIG1,ND1); %data set Clase 1
minr1=min(r1);maxr1=max(r1);

% Generar Datos Clase 2 (d=1)
ND2=400; % Número vectores
mu2 = 7;
SIG2 = 2.5;
r2 = mvnrnd(mu2,SIG2,ND2); %data set Clase 2
minr2=min(r2);maxr2=max(r2);

%Visualización
subplot(2,1,1);
plot(r1, 0.1*ones(ND1),'b+',r2,-0.1*ones(ND2),'ro'); xlabel('x');
axis([min([minr1 minr2]) max([maxr1 maxr2]) -0.2 0.2]);title('Data Set (2 clases, 1d) ')

%% DENSIDAD DE PROBABILIDAD INCONDICIONAL p(x)
R=[r1; r2]; %unificar los datos en un solo vector
minr=min(R);maxr=max(R);
% número bins y particionamiento eje horizontal
%Nbin=(ND1+ND2)/20;
Nbin=(ND1+ND2)/10;

delta=(maxr-minr)/Nbin; %longitud bin
eje=[minr: delta : maxr-delta];
pxa=hist(R,Nbin)/(delta*(ND1+ND2)); 
px = medfilt1(pxa,3); %;%filtro mediana
subplot(2,1,2);
plot(eje,px,'g');title('p(x)');xlabel('x');


