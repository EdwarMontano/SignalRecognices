clc; clearvars ; close all;

%Load Fisher's iris data set. 
load fisheriris
%Remove the sepal lengths and widths, and all observed setosa irises.
inds = ~strcmp(species,'setosa');
X = meas(inds,3:4); % data set -> number of classes=2 (versicolor and virginica); dimension= 2
y = species(inds);  % class label

%  index and total number of patterns by class
dat1= double(strcmp(y(:),'versicolor'));
[indC1]=find(dat1==1); % index: class1
NC1=length(indC1);  %number of patterns: class 1

dat2= double(strcmp(y(:),'virginica'));
[indC2]=find(dat2==1); % index: class2
NC2=length(indC2);%number of patterns: class 2


% Selección porcentajes de entrenamiento y prueba  
Frac = questdlg('Select a Porcentaje Training', '% Training', '60','70','80','60');
Frac=str2num(Frac)/100;

f1=floor(indC1(NC1)*Frac); % End Index: classe 1
f2=NC2+floor(NC2*Frac); % End Index: classe 2

% Select Training dataset
Xtrain= [X(indC1(1):f1 , :) ; X(indC2(1):f2 , :)];
ytrain=  [y(indC1(1):f1) ; y(indC2(1):f2)];

% Select Test dataset
Xtest=  [X(f1+1:indC1(NC1), :) ; X(f2+1: indC2(NC2), :)];
ytest=[ y(f1+1:indC1(NC1)) ; y(f2+1: indC2(NC2))];


% Seleccionar Kernel
kernel = questdlg('Select a Kernel', 'Kernels', 'linear','polynomial','rbf','linear');

% Seleccionar Optimización
Solut= questdlg('Select an Optimization Routine Learning','Optimization','ISDA', 'L1QP','SMO','ISDA');

% Seleccionar remoción de Duplicates
answer1 = questdlg('Would you Remove the Duplicates?', ...
	'Duplicates Remove', 'Yes','No','No');
if strcmp(answer1,'Yes')
    remdup=true;
else
    remdup=false;
end

answer = inputdlg({'Enter the Number Fold'},'Fold',[1 40],{'5'});
nfold= str2num(answer{1});

% Seleccionar Matriz de Costo y Prioridades a Priori 
dlgtitle = 'Costo  - a priori';
dims = [1 35];
definput = {'0 1; 1 0', '0.5 0.5'};
answer = inputdlg({'Matriz Costo', 'Prob. A Priori'},dlgtitle,dims,definput);

Costo=str2num(answer{1});
APriori=str2num(answer{2});


% Definir Número de Folds y Optimización
NObserv=floor((NC1+NC2)*Frac); %Number of Observations
c = cvpartition(NObserv,'KFold',nfold);

opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
        'AcquisitionFunctionName','expected-improvement-plus');

% Train an SVM classifier using the processed data set.
switch kernel
    case 'linear'
        
        SVMModel = fitcsvm(Xtrain,ytrain,'RemoveDuplicates',remdup,'Cost',Costo, 'Prior', APriori , 'KernelFunction', kernel, 'Solver', Solut,...
        'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts)
        
       
    case 'polynomial'
        prompt = {'Enter polinomial order'};
        order = inputdlg(prompt,'Order',[1 50],{'2'});
        order=str2num(order{1});
               
        SVMModel = fitcsvm(Xtrain,ytrain,'RemoveDuplicates',remdup,'Cost',Costo, 'Prior', APriori , 'KernelFunction', kernel,'PolynomialOrder',order , 'Solver', Solut,...
        'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts)
        
        
    case 'rbf'
        
        SVMModel = fitcsvm(Xtrain,ytrain,'RemoveDuplicates',remdup,'Cost',Costo, 'Prior', APriori , 'KernelFunction', kernel, 'Solver', Solut,...
        'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts)
        
    otherwise
         msgbox('Invalid Optimization Routine', 'Error','error');    
end


%The first class ('versicolor') is the negative class, 
% and the second ('virginica') is the positive class. 



%% Plot the observations and the decision boundary. Flag the support vectors
sv = SVMModel.SupportVectors;  

h = 0.02; % Mesh grid step size

[X1,X2] = meshgrid(min(Xtrain(:,1)):h:max(Xtrain(:,1)),...
    min(Xtrain(:,2)):h:max(Xtrain(:,2)));

[~,score_t] = predict(SVMModel,[X1(:),X2(:)]); %Score with trainning dataset

scoreGrid = reshape(score_t(:,1),size(X1,1),size(X2,2));
%scoreGrid = reshape(score_t(:,2),size(X1,1),size(X2,2));


figure
gscatter(Xtrain(:,1),Xtrain(:,2),ytrain, ['r'; 'b'],['+'; 'x'])

hold on
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10)

contour(X1,X2,scoreGrid)

colorbar;
title('{\bf Iris Outlier Detection via Two-Class SVM}')
xlabel('Petal Length (cm)')
ylabel('Petal Width (cm)')
legend('Versicolor','Virginica','Support Vector')
hold off


%% Plot a scatter diagram of the data and circle the support vectors.
sv = SVMModel.SupportVectors;
% Display DataSet Trainning
figure
gscatter(Xtrain(:,1),Xtrain(:,2),ytrain, ['r'; 'b'],['+'; 'x'])
title('Fisher iris - SVM '); xlabel('petal length'); ylabel('petal width');

hold on
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10)% Mark the support vector

%Clasificación --> Dataset Test 
[label,score] = predict(SVMModel,Xtest);

% Display DataSet Test
hold on
gscatter(Xtest(:,1),Xtest(:,2),ytest, ['m'; 'c'], ['+'; 'x'] )

% Display classification result
hold on
gscatter(Xtest(:,1),Xtest(:,2),label, ['m'; 'c'], ['o'; 'o'],10  )

legend('versicolorTrain','virginicaTrain','Support Vector','versicolorTest','virginicaTest','versicolorPred','virginicaPred' )

% Confusion Matrix
rc1=double(strcmp(ytest,'versicolor'));
rc2=double(strcmp(ytest,'virginica'));
Target=[rc1'; rc2'];

rc1=double(strcmp(label,'versicolor'));
rc2=double(strcmp(label,'virginica'));
Estimat=[rc1'; rc2'];

figure
plotconfusion(Target,Estimat); title("Tests Data Set")





