% Maximum Likelihood Parameter Estimation of Gaussian pdfs

% One problem oftenmet in practice is that the pdfs describing the statistical distributionof the data in the
% classes are not known and must be estimated using the training data set.

clear all; close all; clc;

load('iris_dataset');

N =50;
porcentaje_train = 50;
N_datos = round(N*porcentaje_train/100); %los datos de entrenamiento sern los primeros N_datos
c1=3;c2=4; % se eligen 2 de las 4 caractersticas de ficheriris que se usarn para el ejemplo

% Se calcula la media y la varianza de los datos de entrenamient para encontrar outliers
 m = [mean(irisInputs(c1,1:N_datos)) mean(irisInputs(c1,51:50+N_datos)) mean(irisInputs(c1,101:100+N_datos)); ...
      mean(irisInputs(c2,1:N_datos)) mean(irisInputs(c2,51:50+N_datos)) mean(irisInputs(c2,101:100+N_datos))];
 s = [std(irisInputs(c1,1:N_datos)) std(irisInputs(c1,51:50+N_datos)) std(irisInputs(c1,101:100+N_datos)); ...
      std(irisInputs(c2,1:N_datos)) std(irisInputs(c2,51:50+N_datos)) std(irisInputs(c2,101:100+N_datos))];
 % se define el mltiplo de las desviaciones estandar a partir del cuales los datos
 % se consideran outliers
 veces = 2;
 d = veces*s;
 
% Se aplica el criterio a los datos de entrenamiento
X1=[irisInputs(c1,1:N_datos) irisInputs(c1,51:50+N_datos) irisInputs(c1,101:100+N_datos)];
X2=[irisInputs(c2,1:N_datos) irisInputs(c2,51:50+N_datos) irisInputs(c2,101:100+N_datos)];
X_datos =[X1;X2];

class = ones(1,150);class(1,51:100)=2;class(1,101:end)=3;
C = [class(1,1:N_datos) class(1,51:50+N_datos) class(1,101:100+N_datos)];
scatter(X1,X2,N_datos,C,'marker','*'); %dibuja asterscos por cada dato, con un color dependiendo de la clase
hold on;grid on;

 for c=1:3
     plot(m(1,c), m(2,c),'r*'); % dibuja un asterico en el centroide de los datos de entrenamiento
     h = rectangle('position',[m(:,c)'-d(:,c)',2*d(1,c),2*d(2,c)], 'curvature',[0 0]);
        set(h,'linestyle',':') % dibuja un rectngulo hasta donde se cumle la condicin de las desviaciones estndar
 end
 title(strcat('Datos con  ',num2str(veces),' veces la desviacin estndar'));

 k = 1; num_outliers = 0;
 fin = size(X_datos,2);
 for n=1:fin
     if n<N_datos+1
         c=1;
     end
     if and(n >= N_datos+1, n < 2*N_datos+1)
             c=2;
     end
     if n >= 2*N_datos+1
         c=3;
     end
     outlier = sum(abs(X_datos(:,n)-m(:,c))> veces*s(:,c)); %se identifica si se trata de un outlier
     if outlier == 0 %caso no es outlier, se marca con un crculo verde para identificarlo grficamente
         X_train(:,k) = X_datos(:,n); %se adiciona al vector de datos a ser usados
         C2(k) = C(n); 
         k=k+1;
         plot(X_datos(1,n),X_datos(2,n),'go');
     else % caso es un outlier, se marca con un crculo rojo para identificarlo grficamente, y no se adiciona a los datos a ser usados
         disp(strcat('outlier encontrado, dato ',num2str(n)));
         plot(X_datos(1,n),X_datos(2,n),'ro');
         num_outliers = num_outliers+1;
     end
 end
 disp(strcat('se encontraron ',num2str(num_outliers), ' outliers'));
 


