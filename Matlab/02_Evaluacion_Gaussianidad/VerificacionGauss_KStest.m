clc; clearvars; close all; close all hidden

% Kolmogorov-Smirnov Test

% Load Data Set 
% load fisheriris
meas = csvread('../Dataset/Train_Input.csv');
species = csvread('../Dataset/Train_Targets.csv');
[N,d]=size(meas); % N Data numbre number of data the Data-Set, d dimension 
N1=N/3; N2=N/3; N3=N/3; % Data Number for  every class
M=3;  % Number Class

dataset{1}=meas(1:N1, :); % Class 1 data-set
dataset{2}=meas(N1+1:N1+N2, :);% Class 2 data-set
dataset{3}=meas(N1+N2+1:N, :); % Class 3 data-set

% Significance Level 
dat= inputdlg({'Significance level'},'KS Test',1,{'0.01'});
alpha=str2double(dat{1}); % 

% Test the null hypothesis : sample data comes from a normal distribution 
% at the alpha significance level.  

H=zeros(M,d); 
for k=1:M   % Number Class
    X=dataset{1,k} ; % clase k data-set 
    for i=1:d  % Number of caracteristics
        x=X(:,i);
        
       %Create a normal distribution object with parameter values mu and sigma of the dataset caracteristic.
        mu1=mean(x); sigm1=std(x);
        test_pdf = makedist('Normal','mu',mu1,'sigma',sigm1);
        
        %Tests 
        [h,p] = kstest(x,'Alpha',alpha, 'CDF',test_pdf); %Kolmogorov-Smirnov test: h=0 gaussian ,  h=1 NO gaussian
        H(k,i)=h;
    end    
end

% Display test outcomes
f = figure('Position',[440 500 461 146],'Name',['KS Test  alpha=' num2str(alpha)] );
% Create the column and row names in cell arrays 
cnames = {'x1','x2','x3', 'x4'};
rnames = {'Class1','Class2','Class3'};
% Create the uitable
t = uitable(f, 'Data',H, 'ColumnName',cnames,'RowName',rnames); 
% Set width and height
t.Position(3) = t.Extent(3);
t.Position(4) = t.Extent(4);
