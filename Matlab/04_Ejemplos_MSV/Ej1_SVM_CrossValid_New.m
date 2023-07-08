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
Solut= questdlg('Select an Optimization Routine','Optimization','ISDA', 'L1QP','SMO','ISDA');

% Seleccionar número de Folds (Recomendado 5 o 10. Sin embargo, depende del
% tamaño del dataset de entrenamiento. Puede cambiar
dlgtitle = 'kFolds';
dims = [1 35];
definput = {'5'};
answer = inputdlg({'Número de Folds'},dlgtitle,dims,definput);
KF=str2num(answer{1});

% Train an SVM classifier using trainning data set.
switch kernel
    case 'linear'
        CVSVMModel = fitcsvm(Xtrain,ytrain,'CrossVal','on', 'KFold',KF , 'KernelFunction', kernel, 'Solver', Solut);        
       
    case 'polynomial'
        prompt = {'Enter polinomial order'};
        order = inputdlg(prompt,'Order',[1 50],{'2'});
        order=str2num(order{1});
        CVSVMModel  = fitcsvm(Xtrain,ytrain,'CrossVal','on', 'KFold',KF , 'KernelFunction', kernel,'PolynomialOrder',order ,'Solver', Solut);

    case 'rbf'
       CVSVMModel  = fitcsvm(Xtrain,ytrain, 'CrossVal','on', 'KFold',KF ,'KernelFunction', kernel, 'Solver', Solut);
        
    otherwise
         msgbox('Invalid Optimization Routine', 'Error','error');    
end


for k=1: KF
    
    SVMModel=CVSVMModel.Trained{k,1};
    % Plot the observations and the decision boundary. Flag the support vectors
    sv = SVMModel.SupportVectors;  
    
    h = 0.02; % Mesh grid step size
    [X1,X2] = meshgrid(min(Xtrain(:,1)):h:max(Xtrain(:,1)),...
        min(Xtrain(:,2)):h:max(Xtrain(:,2)));
    
    %%  SVM with trainning dataset
    [~,score_t] = predict(SVMModel,[X1(:),X2(:)]); %Score with trainning dataset

    scoreGrid = reshape(score_t(:,1),size(X1,1),size(X2,2));
    %scoreGrid = reshape(score_t(:,2),size(X1,1),size(X2,2));

    figure; subplot(1,2,1);
    gscatter(Xtrain(:,1),Xtrain(:,2),ytrain, ['r'; 'b'],['+'; 'x']);

    hold on
    plot(sv(:,1),sv(:,2),'k d','MarkerSize',10)

    contour(X1,X2,scoreGrid)

    colorbar;
    title(['{\bf Data Set Training Dispertion- SupportVector},  FOLD=' num2str(k)])
    xlabel('Petal Length (cm)')
    ylabel('Petal Width (cm)')
    legend('Versicolor','Virginica','Support Vector')
    hold off


    %% Plot a scatter diagram of the data and circle the support vectors.
    % sv = SVMModel.SupportVectors;
    % Display DataSet Trainning
    
    %figure
    subplot(1,2,2);
    
    gscatter(Xtrain(:,1),Xtrain(:,2),ytrain, ['r'; 'b'],['+'; 'x'])
    title(['Fisher iris - SVM, FOLD=' num2str(k)]); xlabel('petal length'); ylabel('petal width');

    hold on
    plot(sv(:,1),sv(:,2),'k d','MarkerSize',10)% Mark the support vector

    %% Clasificación --> Dataset Test 
    [label,score] = predict(SVMModel,Xtest);

    % Display DataSet Test
    hold on
    gscatter(Xtest(:,1),Xtest(:,2),ytest, ['m'; 'c'], ['+'; 'x'] )

    % Display classification result
    hold on
    gscatter(Xtest(:,1),Xtest(:,2),label, ['m'; 'c'], ['o'; 'o'],10  )

    legend('versicolorTrain','virginicaTrain','Support Vector','versicolorTest','virginicaTest','versicolorPred','virginicaPred' )

    %% Confusion Matrix
    rc1=double(strcmp(ytest,'versicolor'));
    rc2=double(strcmp(ytest,'virginica'));
    Target=[rc1'; rc2'];

    rc1=double(strcmp(label,'versicolor'));
    rc2=double(strcmp(label,'virginica'));
    Estimat=[rc1'; rc2'];

    figure
    plotconfusion(Target,Estimat); hold on; title(['Confusion Matrix FOLD=' num2str(k)]);
    
    
    %h=subplot(1,3,3);
    %confusionchart(Target,Estimat); title(['Confusion Matrix FOLD=' num2str(k)]);
    %confusionchart(label,ytest); title(['Confusion Matrix FOLD=' num2str(k)]);
end 



