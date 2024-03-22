% TBD: compare the performance of the meta-classifier when
% trained on Predictions (i.e. the predicted class) instead of
% the classification Scores

% TBD: compare the performance of the meta-classifier when the
% training split is not performed and the same data is used to 
% train the level-1 classifiers and the meta-classifier.
% Otherwise, you could use the same split but just use the fold1.

%% load dataset
load dataset.mat

%% plot data
u=find(labels_tr==1);
figure(1),hold on
plot(data_tr(u,1),data_tr(u,2),'r.')
u=find(labels_tr==2);
plot(data_tr(u,1),data_tr(u,2),'b.')
hold off

%% stratified sampling
%we are just creating two folds from training set, with n/2 elements
rng('default'); % for reproducibility
idx_f1=[];
idx_f2=[];
for nclass=1:2
    u=find(labels_tr==nclass);
    idx=randperm(numel(u));
    idx_f1=[idx_f1; u(idx(1:round(numel(idx)/2)))];
    idx_f2=[idx_f2; u(idx(1+round(numel(idx)/2):end))];
end
labels_f1=labels_tr(idx_f1);
labels_f2=labels_tr(idx_f2);
data_f1=data_tr(idx_f1,:);
data_f2=data_tr(idx_f2,:);

%% train level-1 classifiers on fold1
mdl={};

% SVM with gaussian kernel
rng('default');
mdl{1}=fitcsvm(data_f1, labels_f1, 'KernelFunction', ...
    'gaussian', 'KernelScale', 5);

% SVM with polynomial kernel
rng('default');
mdl{2}=fitcsvm(data_f1, labels_f1, 'KernelFunction', ...
    'polynomial', 'KernelScale', 10);

% decision tree
rng('default');
mdl{3}=fitctree(data_f1, labels_f1, 'SplitCriterion', ...
    'gdi','MaxNumSplits',20);

% Naive Bayes
rng('default');
mdl{4}=fitcnb(data_f1, labels_f1);

% Ensemble of decision trees
rng('default');
mdl{5}=fitcensemble(data_f1, labels_f1);

%% make the predictions on fold2 (to be used to train the meta-learner)
N=numel(mdl);
Predictions=zeros(size(data_f2,1),N);
Scores=zeros(size(data_f2,1),N);
for ii=1:N 
    [predictions, scores]=predict(mdl{ii},data_f2);
    Predictions(:,ii)=predictions;
    Scores(:,ii)=scores(:,1);
end
    
%% train the stacked classifier on fold2
% predictions of the level-1 classifiers
% train on Scores
rng('default');
stackedModel_Scores = fitcensemble(Scores, labels_f2, 'Method',...
    'Bag');

% stackedModel = fitcensemble(Scores, labels_f2, 'Method',...
%     'AdaBoostM1'); % 'NumLearningCycles'

%train on Predictions
stackedModel_Predictions = fitcensemble(Predictions, labels_f2, 'Method',...
    'Bag');

mdl{N+1}=stackedModel_Scores;
mdl{N+2}=stackedModel_Predictions;

ACC=[];
Predictions=zeros(size(data_te,1),N);
Scores=zeros(size(data_te,1),N);
for ii=1:N
    [predictions, scores]=predict(mdl{ii},data_te);
    Predictions(:,ii)=predictions;
    Scores(:,ii)=scores(:,1);
    ACC(ii)=numel(find(predictions==labels_te))/numel(labels_te);
end

% predictions of the stacked classifier
%predictions = predict(mdl{N+1}, Scores); it shoul be tested as follows
predictions = predict(mdl{N+1}, Scores);
ACC(N+1)=numel(find(predictions==labels_te))/numel(labels_te);

predictions = predict(mdl{N+2}, Predictions);
ACC(N+2)=numel(find(predictions==labels_te))/numel(labels_te);

ACC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%TASK2 
%Compare the performance of the meta-classifier when the training split 
% is not performed and the same data is used to train the level-1 
% classifiers and the meta-classifier.

%% train level-1 classifiers on all training data
mdl={};

% SVM with gaussian kernel
rng('default');
mdl{1}=fitcsvm(data_tr, labels_tr, 'KernelFunction', ...
    'gaussian', 'KernelScale', 5);

% SVM with polynomial kernel
rng('default');
mdl{2}=fitcsvm(data_tr, labels_tr, 'KernelFunction', ...
    'polynomial', 'KernelScale', 10);

% decision tree
rng('default');
mdl{3}=fitctree(data_tr, labels_tr, 'SplitCriterion', ...
    'gdi','MaxNumSplits',20);

% Naive Bayes
rng('default');
mdl{4}=fitcnb(data_tr, labels_tr);

% Ensemble of decision trees
rng('default');
mdl{5}=fitcensemble(data_tr, labels_tr);

%% make the predictions on all training data
%but in this way all the model will be overfitted
N=numel(mdl);
Predictions=zeros(size(data_tr,1),N);
Scores=zeros(size(data_tr,1),N);
for ii=1:N 
    [predictions, scores]=predict(mdl{ii},data_tr);
    Predictions(:,ii)=predictions;
    Scores(:,ii)=scores(:,1);
end

%% train the stacked classifier on training set
% predictions of the level-1 classifiers
% train on Scores
rng('default');
stackedModel_Scores = fitcensemble(Scores, labels_tr, 'Method',...
    'Bag');

% stackedModel = fitcensemble(Scores, labels_f2, 'Method',...
%     'AdaBoostM1'); % 'NumLearningCycles'

%train on Predictions
stackedModel_Predictions = fitcensemble(Predictions, labels_tr, 'Method',...
    'Bag');

mdl{N+1}=stackedModel_Scores;
mdl{N+2}=stackedModel_Predictions;

ACC=[];
Predictions=zeros(size(data_te,1),N);
Scores=zeros(size(data_te,1),N);
for ii=1:N
    [predictions, scores]=predict(mdl{ii},data_te);
    Predictions(:,ii)=predictions;
    Scores(:,ii)=scores(:,1);
    ACC(ii)=numel(find(predictions==labels_te))/numel(labels_te);
end

% predictions of the stacked classifier
predictions = predict(mdl{N+1}, Scores);
ACC(N+1)=numel(find(predictions==labels_te))/numel(labels_te);

predictions = predict(mdl{N+2}, Predictions);
ACC(N+2)=numel(find(predictions==labels_te))/numel(labels_te);

ACC