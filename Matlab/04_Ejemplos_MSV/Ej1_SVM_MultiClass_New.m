clc; clearvars ; close all;

load fisheriris
X = meas(:,3:4);
y = species;
rng(1); % For reproducibility

%%  index and total number of patterns by class

dat1= double(strcmp(y(:),'setosa'));
[indC1]=find(dat1==1); % index: class1
NC1=length(indC1);  %number of patterns: class 1

dat2= double(strcmp(y(:),'versicolor'));
[indC2]=find(dat2==1); % index: class2
NC2=length(indC2);%number of patterns: class 2

dat3= double(strcmp(y(:),'virginica'));
[indC3]=find(dat3==1); % index: class3
NC3=length(indC3);%number of patterns: class 3

% Selección porcentajes de entrenamiento y prueba  
Frac = questdlg('Select a Porcentaje Training', '% Training', '60','70','80','60');
Frac=str2num(Frac)/100;

% End index for each class
%f1=floor(indC1(NC1)*Frac); % End Index: class 1
f1=floor(NC1*Frac); % End Index: class 1
f2=NC2+floor(NC2*Frac); % End Index: class 2
f3=NC1+NC2+floor(NC3*Frac); % End Index: class 3



% Select Training dataset
Xtrain= [X(indC1(1):f1,:) ; X(indC2(1):f2 ,:) ; X(indC3(1):f3,:) ];

ytrain=  [y(indC1(1):f1) ; y(indC2(1):f2); y(indC3(1):f3) ];

% Select Test dataset
Xtest=  [X(f1+1:indC1(NC1), :) ; X(f2+1: indC2(NC2), :) ; X(f3+1: indC3(NC3), :) ];
ytest=[ y(f1+1:indC1(NC1)) ; y(f2+1: indC2(NC2));  y(f3+1: indC3(NC3))  ];


%% TRAIN 
% Kernel selection
KERNEL = questdlg('Select a Kernel', 'Kernels', 'linear','polynomial','rbf','linear');

% Coding Matrix selection
lista={'onevsone','onevsall','binarycomplete', 'denserandom', ...
       'ordinal' , 'sparserandom','ternarycomplete'};

[indCM,~] = listdlg('PromptString',{'Select a Coding Matrix'},...
        'SelectionMode','single','ListString',lista);
CodMat=lista{indCM};
            
% Create an SVM template (Most of its properties are empty). 
% Standardize the predictors, and specify the selectioned kernel.

templ = templateSVM('Standardize',true,'KernelFunction',KERNEL);

%Train the ECOC classifier using the SVM template. 
Mdl = fitcecoc(Xtrain,ytrain,'Learners',templ,'FitPosterior',true,...
    'ClassNames',{'setosa','versicolor','virginica'},'Coding',CodMat, 'Verbose',2);

% Display Coding Matrix
CodingMat = Mdl.CodingMatrix

% Predict the TRAINING-SAMPLE labels and class posterior probabilities. 
% Display diagnostic messages during the computation of labels 
% and class posterior probabilities by using the 'Verbose' name-value pair argument.

[labelTrain,~,~,PosteriorTrain] = resubPredict(Mdl,'Verbose',1);

%Display a random set of results.
idx = randsample(size(Xtrain,1),10,1);

% Display Table
% The columns of Posterior correspond to the class order of Mdl.ClassNames.
%table(Y(idx),label(idx),Posterior(idx,:),...
T1=table(ytrain(idx),labelTrain(idx),PosteriorTrain(idx,:),...
    'VariableNames',{'TrueLabel','PredLabel','Posterior'})

%% TEST Data-set

% Predict the TEST-SAMPLE labels and class posterior probabilities. 
% Display diagnostic messages during the computation of labels 
% and class posterior probabilities by using the 'Verbose' name-value pair argument.

[labelTest,~,~,PosteriorTest] = predict(Mdl,Xtest);

%Display a random set of results.
idx = randsample(size(Xtest,1),10,1);

% Display Table
% The columns of Posterior correspond to the class order of Mdl.ClassNames.
T2=table(ytest(idx),labelTest(idx),PosteriorTest(idx,:),...
    'VariableNames',{'TrueLabel','PredLabel','Posterior'})

%% DISPLAY : Define a grid of values in the observed predictor space. 
%   Predict the posterior probabilities for each instance in the grid.
xMax = max(Xtest);
xMin = min(Xtest);

x1Pts = linspace(xMin(1),xMax(1));
x2Pts = linspace(xMin(2),xMax(2));
[x1Grid,x2Grid] = meshgrid(x1Pts,x2Pts);

[~,~,~,PosteriorRegion] = predict(Mdl,[x1Grid(:),x2Grid(:)]);

%For each coordinate on the grid, plot the maximum class posterior probability among all classes.

contourf(x1Grid,x2Grid,...
        reshape(max(PosteriorRegion,[],2),size(x1Grid,1),size(x1Grid,2)));
h = colorbar; 
h.YLabel.String = 'Maximum posterior';
h.YLabel.FontSize = 15;

hold on
gh = gscatter(Xtest(:,1),Xtest(:,2),ytest,'krk','*xo',6); %Display test-dataset observations
gh(2).LineWidth = 2;
gh(3).LineWidth = 2;

title('Iris Petal Measurements and Maximum Posterior')
xlabel('Petal length (cm)')
ylabel('Petal width (cm)')
axis tight
legend(gh,'Location','NorthWest')
hold off

%% Confusion Matrix
rct1=double(strcmp(ytest,'setosa'));
rct2=double(strcmp(ytest,'versicolor'));
rct3=double(strcmp(ytest,'virginica'));
Target=[rct1'; rct2'; rct3'];

rcp1=double(strcmp(labelTest,'setosa'));
rcp2=double(strcmp(labelTest,'versicolor'));
rcp3=double(strcmp(labelTest,'virginica'));
Estimat=[rcp1'; rcp2'; rcp3' ];

figure
plotconfusion(Target,Estimat)
title('Confusion Matrix  - Test DataSet');
