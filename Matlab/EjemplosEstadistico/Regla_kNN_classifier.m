close 'all';
clearvars;
clc;
% 1 Generate datasets X1 and X2
M=2; %Number classes
m1=[0; 0];  m2=[1; 2];
m=[m1  m2];
S1=[0.8 0.2;0.2 0.8];
S2=[0.8 0.2;0.2 0.8];
S(:,:,1)=S1;S(:,:,2)=S2;
P=[1/2 1/2 ]'; N1=1000;
randn('seed',0);
[X1,y1]=generate_gauss_classes(m,S,P,N1);
N2=5000;
randn('seed',100);
[X2,y2]=generate_gauss_classes(m,S,P,N2);

% 2 - Classification X2 with k_nn_classifier 
%knn=3;
knn=5; 
%knn=7;
%knn=11;
AssigClass=k_nn_classifier(X1,y1,knn,X2);

%% 3. Compute the confusion matrix 
% Index set for plotconfusion
% Labe Tests Data Set 
ind_t1=find(y2==1);ind_t2=find(y2==2); 
Targets=zeros(M,N2);
Targets(1,ind_t1)=1; Targets(2,ind_t2)=1; 

%% Label kNN Classifier 
ind_1=find(AssigClass==1); ind_2=find(AssigClass==2); 
Outputs=zeros(M,N2);
Outputs(1,ind_1)=1; Outputs(2,ind_2)=1; 
% Plot Confusion
figure; plotconfusion(Targets,Outputs,['kNN Classifier, knn=' num2str(knn)  ]) 

