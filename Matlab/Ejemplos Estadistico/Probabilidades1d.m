clc; clearvars ; close all;
% Generar Datos Clase 1 (d=1)
ND1=400; % Número vectores 
mu1 = 3; % Promedio
SIG1 = 1.5; % Varianza
r1 = mvnrnd(mu1,SIG1,ND1); %data set Clase 1
minr1=min(r1);maxr1=max(r1);

% Generar Datos Clase 2 (d=1)
ND2=400; % Número vectores
mu2 = 7;
SIG2 = 2.5;
r2 = mvnrnd(mu2,SIG2,ND2); %data set Clase 2
minr2=min(r2);maxr2=max(r2);

%Probabilidad A Priori de Clases
Pw1=0.5; Pw2=0.5; 

%Visualización
subplot(3,2,1);
plot(r1, 0.1*ones(ND1),'b+',r2,-0.1*ones(ND2),'ro')
axis([min([minr1 minr2]) max([maxr1 maxr2]) -0.2 0.2])

%% DENSIDADES DE PROBABILIDAD CONDICIONAL DE CLASE p(x|w)
%p(x|w1)
% número bins y particionamiento eje horizontal
%Nbin=ND1/20;
Nbin=ND1/10;
delta1=(maxr1-minr1)/Nbin; %longitud bin
eje1=[minr1: delta1 : maxr1-delta1];
pxw1a=hist(r1,Nbin)/(delta1*ND1); % Se divide por delta para obtener la densidad de probabilidad
%pxw1 = medfilt1(pxw1a,3); %;%filtro mediana
pxw1 =pxw1a;
subplot(3,2,3);
plot(eje1,pxw1,'b');title('p(x|w1)');

%p(x|w2)
% número bins y particionamiento eje horizontal
%Nbin=ND2/20;
Nbin=ND2/10;
delta2=(maxr2-minr2)/Nbin;%longitud bin
eje2=[minr2: delta2 : maxr2-delta2];
pxw2a=hist(r2,Nbin)/(delta2*ND2);
%pxw2 = medfilt1(pxw2a,3);
pxw2=pxw2a;
subplot(3,2,4); 
plot(eje2,pxw2,'r');title('p(x|w2)');

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
subplot(3,2,2);
plot(eje,px,'g');title('p(x)');

%% DENSIDAD DE PROBABILIDAD CONJUNTA p(w,x)= p(x|w) P(w)
%Pw1=0.5; Pw2=0.5; %Probabilidad A Priori
%p(w1,x)
pw1x=pxw1*Pw1;
subplot(3,2,5);
plot(eje1,pw1x,'b');title('p(w1,x)');
%p(w2,x)
pw2x=pxw2*Pw2;
subplot(3,2,6);
plot(eje2,pw2x,'r');title('p(w2,x)');

%% Generación de Datos de Prueba Test

% Generar Datos Clase 1 (d=1)
NT1=50; % Número vectores 
mu1 = 3; SIG1 = 1.5;
t1 = mvnrnd(mu1,SIG1,NT1); %Test data set Clase 1
% Generar Datos Clase 2 (d=1)
NT2=50; % Número vectores
mu2 = 7; SIG2 = 2.5;
t2 = mvnrnd(mu2,SIG2,NT2); %Test data set Clase 2

%% Asignación Clasificador Bayesiano (no se tiene en cuenta el denominador p(x))
%% 

T=[t1; t2]; % Test dataset 

%Obtener 
clase=[];
for i=1:length(T)
    x= T(i);
    %obtener P(w1|x); a priori Probability 
    if x<minr1 || x>maxr1 
        APwx1=0;
    else
        ind1=ceil((x-minr1)/delta1);
        APwx1=pw1x(ind1);
    end    
    
    %obtener P(w2|x)
    if x<minr2 || x>maxr2 
        APwx2=0;
    else
        ind2=ceil((T(i)-minr2)/delta2);
        APwx2=pw2x(ind2);
    end    
    
    % Clasificar
    
    if APwx1>APwx2
       EClase(i)=1;        
    else
       EClase(i)=2; 
    end    
end

% Graficación

figure
subplot(2,1,1); plot(T, zeros(length(T)),'cd');title('Test Dataset'),xlabel('x');
n1=1:length(t1); n2=length(t1)+1:length(T);
subplot(2,1,2); plot(T(n1), EClase(n1),'b+',T(n2), EClase(n2),'ro');title('Clasification'),xlabel('x');











