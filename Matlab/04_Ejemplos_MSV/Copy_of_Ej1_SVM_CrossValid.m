clc; clearvars ; close all;

%Load Fisher's iris data set. 
% load fisheriris
meas = csvread('../Dataset/Train_Input.csv');
species = readcell('../Dataset/Train_TargetsName.csv');
%Remove the sepal lengths and widths, and all observed setosa irises.
inds = ~strcmp(species,'medicamento1');
X = meas(inds,1:2); % data set -> number of classes=2 (versicolor and virginica); dimension= 2
y = species(inds);  % class label

%  index and total number of patterns by class
dat1= double(strcmp(y(:),'medicamento2'));
[indC1]=find(dat1==1); % index: class1
NC1=length(indC1);  %number of patterns: class 1

dat2= double(strcmp(y(:),'medicamento3'));
[indC2]=find(dat2==1); % index: class2
NC2=length(indC2);%number of patterns: class 2


% Seleccin porcentajes de entrenamiento y prueba  
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

% Seleccionar Optimizacin
Solut= questdlg('Select an Optimization Routine','Optimization','ISDA', 'L1QP','SMO','ISDA');

% Train an SVM classifier using the processed data set.
switch kernel
    case 'linear'
        SVMModel = fitcsvm(Xtrain,ytrain,'CrossVal','on', 'KFold',4 , 'KernelFunction', kernel, 'Solver', Solut);        
        
        
    case 'polynomial'
        prompt = {'Enter polinomial order'};
        order = inputdlg(prompt,'Order',[1 50],{'2'});
        order=str2num(order{1});
        SVMModel = fitcsvm(Xtrain,ytrain, 'KernelFunction', kernel,'PolynomialOrder',order ,'Solver', Solut);

    case 'rbf'
       % SVMModel = fitcsvm(Xtrain,ytrain, 'KernelFunction', kernel, 'Solver', Solut);
       NObserv=floor((NC1+NC2)*Frac); %Number of Observations
        c = cvpartition(NObserv,'KFold',10);
        opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
        'AcquisitionFunctionName','expected-improvement-plus');
        SVMModel = fitcsvm(Xtrain,ytrain, 'KernelFunction', kernel, 'Solver', Solut,...
        'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts)
        
    otherwise
         msgbox('Invalid Optimization Routine', 'Error','error');    
end


% Seleccionar Cross-validate the SVM classifier. By default, the software uses 10-fold cross-validation.
Cross_Val= questdlg('Select Cross-Validate Option','Cross-Validate','Yes', 'No', 'Yes');
if strcmp(Cross_Val,'Yes')
    CvSVMModel = crossval(SVMModel, 'KFold',10);
end    

%Display the properties of SVMModel. For example, to determine the class order, use dot notation.

%classOrder = SVMModel.ClassNames
%classOrder = CvSVMModel.ClassNames;

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
gscatter(Xtrain(:,1),Xtrain(:,2),ytrain, 'rb','+x')

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
gscatter(Xtrain(:,1),Xtrain(:,2),ytrain, 'rb','+x')
title('Fisher iris - SVM '); xlabel('petal length'); ylabel('petal width');

hold on
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10)% Mark the support vector

%Clasificacin --> Dataset Test 
[label,score] = predict(SVMModel,Xtest);

% Display DataSet Test
hold on
gscatter(Xtest(:,1),Xtest(:,2),ytest, 'mc', '+x' )

% Display classification result
hold on
gscatter(Xtest(:,1),Xtest(:,2),label, 'mc', 'oo',10  )

legend('versicolorTrain','virginicaTrain','Support Vector','versicolorTest','virginicaTest','versicolorPred','virginicaPred' )

% Confusion Matrix
rc1=double(strcmp(ytest,'medicamento1'));
rc2=double(strcmp(ytest,'medicamento2'));
Target=[rc1'; rc2'];

rc1=double(strcmp(label,'medicamento1'));
rc2=double(strcmp(label,'medicamento2'));
Estimat=[rc1'; rc2'];

figure
plotconfusion(Target,Estimat)





