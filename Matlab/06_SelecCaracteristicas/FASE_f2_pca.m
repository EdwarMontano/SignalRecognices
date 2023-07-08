load('digitos_train.mat');
load('digitos_test.mat');
% Xtrain, Ltrain, Xtest, Ltest
displ = 1;

% ejemplos de los datos representados como imagenes
if displ
    figure;
    for i=1:9
        subplot(3,3,i)
        image(Xtrain(:,:,i))
        colormap gray(255)
        title(['Imagen: ' num2str(i) ', digito: ' num2str(Ltrain(i))])
    end

    % imagen original
    figure;
    for i=1:9
        j = i+9;
        subplot(3,3,i)
        image(Xtrain(:,:,j))
        colormap gray(255)
        title(['Imagen: ' num2str(j) ', digito: ' num2str(Ltrain(j))])
    end
end

%% construccion de datos
n = size(Xtrain); 
num = 3;
indexes = Ltrain == num;
labels = Ltrain(indexes);
data = Xtrain(:,:,indexes);

figure;
for i=1:9
    subplot(3,3,i)
    image(data(:,:,i))
    colormap gray(255)
    title(['Imagen: ' num2str(i) ', digito: ' num2str(labels(i))])
end

N = n(1)*n(2); % dimension de caracteristicas (28*28 = 784)
M = length(labels);
disp(['Dimension del problema:  ' num2str(N)])
disp(['Cantidad de datos:  ' num2str(M)])

x_train = reshape(data,N,M);

%% PCA
% matriz de covarianza
covx=cov(x_train');

% numero de vectores propios
num_k = 784;
% V: eigenvectors, D: eigenvalues
[V, D] = eigs(covx,num_k);
D = diag(D);

x_trainPCA = (x_train' * V)';

%% plot de valores propios
[Vfull, Dfull] = eig(covx);
vals = diag(Dfull);
figure;plot(vals(end:-1:1),'k:*')
title('Valores propios')

%% Test
indexes_test = Ltest == num;
labels_test = Ltest(indexes_test);
K = length(labels_test);
xtest = Xtest(:,:,indexes_test);
x_test = reshape(xtest,N,K);

% proyeccion (se pierde informacion, espacio de menor dimension)
x_testPCA  = (x_test' * V)';
% reconstruccion
x_test_reconst = (V*x_testPCA);
% reshape a imagen
x_test_imag = reshape(x_test_reconst,n(1),n(2),K);

figure;
for i=1:9
    subplot(3,3,i)
    image(xtest(:,:,i))
    colormap gray(255)
    title(['Imagen: ' num2str(i) ', digito: ' num2str(labels(i))])
end

figure;
for i=1:9
    subplot(3,3,i)
    image(x_test_imag(:,:,i))
    colormap gray(255)
    title(['Imagen: ' num2str(i) ', digito: ' num2str(labels(i))])
end