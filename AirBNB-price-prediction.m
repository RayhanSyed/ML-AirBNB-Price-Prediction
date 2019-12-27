%{
Columns of features of matrix point ln:
1 - lat
2 - long
3 - rm type
4 - price
5 - min nights 
6 - # of reviews
7 - rev/ month
8 - cal host listing count
9 - availability /365
10 - neighborhood ID
%}
L = 48895; % starting data point
rng(255) % reproducability

%%
backup = ABNYC1(:,[7 8 9 10 11 12 14 15 16]); %backup
Data = backup; % Data
M = Data{:,:}; % turn to numerical array
M = [M transpose(1:48895)]; % adding a row # ID used for rearrangment later
%%
% Eliminate rows that are missing elements of data
kn = zeros(size(M));
order = zeros(L, 1);
j=1;
for i=1:L
if ~(isnan(M(i,7)))
    kn(j, :) = M(i,:);    
    order(j)= i;
    j=j+1;
end
end
kn = kn(1:j-1,:);
order = order(1:j-1,:);
%%
% Remove data that has too low of a review # or 0 days of availability 
[B, I] = sort(kn(:,9));
kn = kn(I,:);
start = sum(kn(:,9)==0);
stop = nnz(kn(:,9));
ln = kn(1 + start: start + stop,:);
entries = size(ln);
entries = entries(1);
%%
neighborhoodSpan = zeros(L, 221); %sort through neighborhoods to get Neighborhood identification in a sparse matrix and in an index
neighborhoodID = zeros(1,48895);
neighborhood = categorical(ABNYC1.neighbourhood);
lastN = neighborhood(1);
neighborhoodSpan(1,1) = 1;
neighborhoodID(1) = 1;
j=1;
for i=2:L
    lastN = neighborhood(i-1);
    if(neighborhood(i) == lastN)
    neighborhoodSpan(i,j) = 1;
    neighborhoodID(i) = j;
    else
            j = j+1;
            neighborhoodSpan(i,j) = 1;
            neighborhoodID(i) = j;
    end
end
neighborhoodID = neighborhoodID(ln(:,10));
mn = [ln(:,1:9) transpose(neighborhoodID)];
max(unique(neighborhoodID));
missing = [1:217] - unique(neighborhoodID);
%%
Labels = mn(:,4); %Take subset of data that is used as variables for features or hierachical parameters
Features = mn(:,[1:3 5:10]);
Features = Features(:,[ 1 2 4 5 6 7 8 3 9]);
N = 500;% Test set #
O = entries - N;
randseq = randperm(entries); %Make traing and test set
training = randseq(1:entries-N);
testing = randseq(entries-N+1:entries);
Training_Data = Features(training,:);
Testing_Data = Features(testing,:);
TrainingLabels = Labels(training);
TestLabels = Labels(testing);
%%
iter=64;
Alpha = ones(7, 663); % Hierarchical Matrix with 7 variables across 663 combinations of neighborhood and rm type 
lr = 1e-3; %learning rate
index = Training_Data(:,9)*3 - (3-Training_Data(:,8)); %formula for mapping a given listings's weights
PriceEst=zeros(O, 1);
PriceEstb=zeros(O, 1);
PriceGuess = zeros(N,1);
TrainingLabels;
TestLabels;
for i =1:O
   PriceEst(i) = sum(Training_Data(i,[1:7]) * Alpha(:,index(i)));
end
%%
% Implement gradient descent to get optimal weights for every hierarchy combination
for b = 1:663
        for a=1:7
    j = Alpha(a, b);
    for k=1:iter
        for i =1:O
   PriceEst(i) = sum(Training_Data(i,[1:7]) * Alpha(:,index(i)));
        end
Diff = PriceEst - TrainingLabels;
SqDiff = Diff.^2;
MSE_Train1 = sum(SqDiff)/O;
Alpha(a, b) = j + lr;
    for i=1:O
PriceEstb(i) = sum(Training_Data(i,[1:7]).* transpose(Alpha(:,index(i))));
    end
Diffb = PriceEstb - TrainingLabels;
SqDiffb = Diffb.^2;    
MSE_Train2 = sum(SqDiffb)/O;
MSEDiff = MSE_Train1 - MSE_Train2;
j = j + MSEDiff;
    end
        end
end

%%

% Apply weights to test set and get MSE
index2 = Testing_Data(:,9)*3 - (3-Testing_Data(:,8));
for i=1:N
PriceGuess(i) = abs(sum(Testing_Data(i,[1:7]) .* transpose(Alpha(:,index2(i)))));
end
Diff = PriceGuess - TestLabels;
SqDiff = Diff.^2;
MSE_Test = sum(SqDiff)/N;