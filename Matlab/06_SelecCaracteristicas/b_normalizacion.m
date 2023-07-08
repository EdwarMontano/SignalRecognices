clear all; close all; clc;

load fisheriris;

X=meas(:,3:4);
y=zeros(3,150);
y(1,1:50)=1;
y(2,51:100)=1;
y(3,101:150)=1;
scatter(X(:,1)',X(:,2)',60, y','o', 'fill'); grid on;
title('Datos Originales');
% Clasificacin con rboles de desicin con los datos sin procesar
% Partition the data into training (50%) and validation (50%) sets.

n = size(meas,1);
rng(1) % For reproducibility
idxTrn = false(n,1);
idxTrn(randsample(n,round(0.5*n))) = true; % Training set logical indices
idxVal = idxTrn == false;                  % Validation set logical indices

% Normalizacin media cero desviacin estandar la unidad
m3=mean(X(idxTrn,1)); r3=std(X(idxTrn,1));
m4=mean(X(idxTrn,2)); r4=std(X(idxTrn,2));
% Para todos los datos se usa la media y desviacin encontrada con los
% datos de entrenamiento
xn1(:,1)=(X(:,1)-m3)/r3;
xn1(:,2)=(X(:,2)-m4)/r4;
% % Tambin funciona con la orden en Matlab mapstd(X(idxTrn)')';
figure;
scatter(xn1(:,1)',xn1(:,2)',60, y','o', 'fill'); grid on;
title('Media cero, Desviacin estndar 1');

%Normalizacin 0 a 1
m_1 = min(X(idxTrn,1));m_2 = min(X(idxTrn,2));
M_1 = max(X(idxTrn,1)); M_2 = max(X(idxTrn,2));
% Para todos los datos se usan los valores mnimo mximo encontrados con los
% datos de entrenamiento
xn2(:,1)=(X(:,1)-m_1)/(M_1-m_1);
xn2(:,2)=(X(:,2)-m_2)/(M_2-m_2);
figure;
scatter(xn2(:,1)',xn2(:,2)',60, y','o', 'fill'); grid on;
title('Rango 0 a 1');
% % Tambin funciona con la orden en Matlab mapminmax(X(idxTrn)',0,1)';


%Normalizacin Softmax
% la normalizacin softmax usa la media (m3 y m4) y desviacin estndar (r3 y r4)
% ya calculados a partir de los datos de entrenamiento para la primera normalizacin,
% y se aplica a todos los datos
media(1,:)=m3*ones(1,size(X,1));media(2,:)=m4*ones(1,size(X,1));
desviacion(1,:)=r3*ones(1,size(X,1));desviacion(2,:)=r4*ones(1,size(X,1));
t=-100;
xn4=(1./exp((X'-media)./(t*desviacion)))';

figure;
scatter(xn4(:,1)',xn4(:,2)',60, y','o', 'fill'); grid on;
title(strcat('Softmax con t =',num2str(t)));

