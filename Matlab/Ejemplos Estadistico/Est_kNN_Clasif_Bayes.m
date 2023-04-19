clc; close vars all; clearvars;

%% Parameters for generate the data set 
d=3; %dimensionality
c=3; % quantity classes

m1=[0 0 0]'; %means vectors
m2=[1 2 2]';
m3=[3 3 4]';
m=[m1 m2 m3];

S1=[0.8  0.2  0.1; 0.2  0.8  0.2;  0.1  0.2  0.8]; %covariance matrices 
S2=[0.6  0.01  0.01; 0.01  0.8  0.01;  0.01  0.01  0.6];
S3=[0.6  0.1  0.1; 0.1  0.6  0.1;  0.1  0.1  0.6];
S(:,:,1)=S1;S(:,:,2)=S2;S(:,:,3)=S3;

% A priori probabilities
P1=1/2;  P2=1/4; P3=1/4 ;
P=[P1 P2 P3]';

%% Generate data set training:  Xa (aprendizaje)
N=1000;
N1=fix(N*P1); N2=fix(N*P2); N3=fix( N*P3);
randn('seed',0);
[Xa,ya]=generate_gauss_classes(m,S,P,N);
% Generate data set test: Xp (prueba)
randn('seed',100);
Nt=500; 
Nt1=fix(Nt*P1); Nt2=fix(Nt*P2); Nt3=fix( Nt*P3);
[Xp,yp]=generate_gauss_classes(m,S,P,Nt);

% Display Xe (Entrenamiento) 
subplot(2,1,1); plot3(Xa(1,1:N1),Xa(2,1:N1),Xa(3,1:N1),'*r',Xa(1,N1+1:N1+N2),...
    Xa(2,N1+1:N1+N2),Xa(3,N1+1:N1+N2),'*b', Xa(1,N1+N2+1:N1+N2+N3),...
    Xa(2,N1+N2+1:N1+N2+N3),Xa(3,N1+N2+1:N1+N2+N3),'*k'); 
grid on; title(['Training Data Set, N=' num2str(N)]); 
xlabel(['x1, N1=' num2str(N1)] );ylabel(['x2 N2=' num2str(N2)]);zlabel(['x3, N3=' num2str(N3)]);
% Display Xp (Prueba)
subplot(2,1,2); plot3(Xp(1,1:Nt1),Xp(2,1:Nt1),Xp(3,1:Nt1),'or',Xp(1,Nt1+1:Nt1+Nt2),...
    Xp(2,Nt1+1:Nt1+Nt2),Xp(3,Nt1+1:Nt1+Nt2),'ob', Xp(1,Nt1+Nt2+1:Nt1+Nt2+Nt3),...
    Xp(2,Nt1+Nt2+1:Nt1+Nt2+Nt3),Xp(3,Nt1+Nt2+1:Nt1+Nt2+Nt3),'ok'); 
grid on; title(['Test Data Set, N=' num2str(Nt)]); 
xlabel(['x1, Nt1=' num2str(Nt1)] );ylabel(['x2 Nt2=' num2str(Nt2)]);zlabel(['x3, Nt3=' num2str(Nt3)]);

%% 1. Compute the kNN estimates of the pdfs p^(x|wi)
%knn=20;
knn=15;
px_c1=estimar_kNN_density(Xp,Xa(:,1:N1), knn); % p(x|w1)
px_c2=estimar_kNN_density(Xp,Xa(:,N1+1:N1+N2), knn); % p(x|w2)
px_c3=estimar_kNN_density(Xp,Xa(:,N1+N2+1+1:N1+N2+N3), knn); % p(x|w3)

%% 2. Bayesian classifier, using the kNN pdf estimates, in order to
% classify the data vectors of Xp. Assign x to wi if  p(x|wi)P(wi) > p(x|wj) P(wj)
A_Prob_Xp= [P1*px_c1; P2*px_c2; P3*px_c3;]; % A Posteriori Probability
[MAX, AssigClass]=max(A_Prob_Xp); % 

%% 3. Compute the classification error for each classifier
% Index set for plotconfusion
% Labe Tests Data Set 
ind_t1=find(yp==1);ind_t2=find(yp==2); ind_t3=find(yp==3);
Targets=zeros(c,Nt);
Targets(1,ind_t1)=1; Targets(2,ind_t2)=1; Targets(3,ind_t3)=1;

%% Label Bayes Classifier with kNN pdf estimates
ind_b1=find(AssigClass==1); ind_b2=find(AssigClass==2); ind_b3=find(AssigClass==3);
OutputsB=zeros(c,Nt);
OutputsB(1,ind_b1)=1; OutputsB(2,ind_b2)=1; OutputsB(3,ind_b3)=1;
% Plot Confusion
figure; plotconfusion(Targets,OutputsB,['Bayesian Class, knn=' num2str(knn)  ]) 
