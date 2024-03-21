accuracy_5x2=[];

for ndataset=1:4
    switch ndataset
        case 1, load dataset1.mat
        case 2, load dataset2.mat
        case 3, load dataset3.mat
        case 4, load dataset4.mat
        otherwise
    end 
    
    accuracyTimes = [];
    accuracyTimes_svm_rbf = [];
    accuracyTimes_knn = [];
    accuracyTimes_tree = [];
   
    for ntimes=1:5
            % stratified sampling
            idx_tr=[];
            idx_te=[];
            for nclass=1:2
                u=find(labels==nclass);
                idx=randperm(numel(u));
                idx_tr=[idx_tr; u(idx(1:round(numel(idx)/2)))];
                idx_te=[idx_te; u(idx(1+round(numel(idx)/2):end))];
            end
        
            labels_tr=labels(idx_tr);
            labels_te=labels(idx_te);
            data_tr=data(idx_tr,:);
            data_te=data(idx_te,:);
            
            % Training a classifier 
            % train on training split, test on the test split

            %Linear SVM
            SVM_LIN=fitcsvm(data_tr, labels_tr, 'KernelFunction','linear','KernelScale',1);

            %SVM gaussian
            SVM_RBF=fitcsvm(data_tr, labels_tr, 'KernelFunction','gaussian','KernelScale',0.1);
            
            %KNN
            KNN_Model = fitcknn(data_tr,labels_tr,'Distance','Euclidean','NumNeighbors',10);
    
            %Tree model
            Tree_Model = fitctree(data_tr,labels_tr,'SplitCriterion','gdi','MaxNumSplits', 10);
   
    
            % Make a prediction on the test set
            prediction=predict(SVM_LIN, data_te);
            prediction_svm_rbf=predict(SVM_RBF, data_te);
            prediction_knn=predict(KNN_Model, data_te);
            prediction_tree=predict(Tree_Model, data_te);
            
    
            accuracy1=numel(find(prediction==labels_te))/numel(labels_te);
            accuracy1_svm_rbf=numel(find(prediction_svm_rbf==labels_te))/numel(labels_te);
            accuracy1_knn=numel(find(prediction_knn==labels_te))/numel(labels_te);
            accuracy1_tree=numel(find(prediction_tree==labels_te))/numel(labels_te);
            
    
    
            % reversing the role of training and test:
            SVM_LIN=fitcsvm(data_te, labels_te, 'KernelFunction','linear','KernelScale',1);
            SVM_RBF=fitcsvm(data_te, labels_te, 'KernelFunction','gaussian','KernelScale',0.1);
            KNN_Model = fitcknn(data_te,labels_te,'Distance','Euclidean','NumNeighbors',10);
            Tree_Model = fitctree(data_te,labels_te,'SplitCriterion','gdi','MaxNumSplits', 10);
           
    
            % Make a prediction on the test set

            %Linear SVM prediction
            prediction=predict(SVM_LIN, data_tr);

            %SVM Gaussian prediction
            prediction_svm_rbf=predict(SVM_RBF, data_tr);
    
            %KNN prediction
            prediction_knn=predict(KNN_Model, data_tr);
    
            %Tree prediction
            prediction_tree=predict(Tree_Model, data_tr);
    
    
            accuracy2=numel(find(prediction==labels_tr))/numel(labels_tr);
            accuracy2_svm_rbf=numel(find(prediction_svm_rbf==labels_tr))/numel(labels_tr);
            accuracy2_knn=numel(find(prediction_knn==labels_tr))/numel(labels_tr);
            accuracy2_tree=numel(find(prediction_tree==labels_tr))/numel(labels_tr);
    
            accuracy=(accuracy1+accuracy2)/2;
            accuracy_svm_rbf=(accuracy1_svm_rbf+accuracy2_svm_rbf)/2;
            accuracy_knn = (accuracy1_knn+accuracy2_knn)/2;
            accuracy_tree = (accuracy1_tree + accuracy2_tree) / 2;
    
    
            accuracyTimes(ntimes,1)=accuracy;
            accuracyTimes_svm_rbf(ntimes,1)=accuracy_svm_rbf;
            accuracyTimes_knn(ntimes,1)=accuracy_knn;
            accuracyTimes_tree(ntimes,1)=accuracy_tree;
        
     end
    

    accuracy_5x2(ndataset,1)=mean(accuracyTimes);
    accuracy_5x2(ndataset,2)=mean(accuracyTimes_svm_rbf);    
    accuracy_5x2(ndataset,3)=mean(accuracyTimes_knn);
    accuracy_5x2(ndataset,4)=mean(accuracyTimes_tree);
    

   
end

accuracy_5x2


for i = 1:size(accuracy_5x2, 1)
    [~, sortIdx] = sort(accuracy_5x2(i, :), 'descend');
    ranks(i, sortIdx) = 1:numel(sortIdx);
end


ranks

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% mean ranks for each classifier across all datasets
meanRanks = mean(ranks, 1);

disp('Mean ranks of classifiers across all datasets:');
disp(meanRanks);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%FRIEDMAN TEST

[pValue, tbl, stats] = friedman(1 - accuracy_5x2, 1, 'off');

fprintf('P-value from Friedman test: %.4f\n', pValue);

stats

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classifiers = {'SVM Linear', 'SVM RBF', 'KNN', 'Tree'};
[c,m,h,gnames] = multcompare(stats);


%p-values
for i = 1:size(c, 1)
    fprintf('%s vs %s: p-value = %.4f\n', classifiers{c(i, 1)}, classifiers{c(i, 2)}, c(i, 6));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate the critical difference (CD)
N = 4; 
k = 4; 
alpha = 0.05; 

% Critical Value for the Nemenyi test
% This uses the studentized range statistic (q) for k groups and N datasets.
q_alpha = 2.569; 

CD = q_alpha * sqrt((k*(k+1))/(6*N));

fprintf("\n")
disp(['Critical Difference (CD) at alpha = ' num2str(alpha) ' is: ' num2str(CD)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf("\n")
meanRanks = stats.meanranks;
diffMatrix = abs(meanRanks' - meanRanks);

for i = 1:length(classifiers)
    for j=i+1:length(classifiers)
        diff=diffMatrix(i,j);
        fprintf("%s vs %s: Mean Rank Difference = %.4f", classifiers{i}, classifiers{j}, diff);
        if diff > CD
            fprintf(" Which is higher than the CD of %.4f. \n", CD);
        else
             fprintf(" Which not higher than the CD of %.4f. \n", CD);
        end
    end
 end

