Data_Train = loadMNISTImages('train-images.idx3-ubyte');
Train_Labels = loadMNISTLabels('train-labels.idx1-ubyte');
Data_Test = loadMNISTImages('t10k-images.idx3-ubyte');
Test_Labels = loadMNISTLabels('t10k-labels.idx1-ubyte');

% Apply PCA to data comment it 

% d is number of principal components 
% you can change its number 
d=40;
% careful to remove mean of training data among all dimension from test 
% data, cannot use test data.

mean_data=mean(Data_Train,2);
Data_Train=Data_Train-repmat(mean_data,1,size(Data_Train,2));
Data_Test=Data_Test-repmat(mean_data,1,size(Data_Test,2));

% Find Principal Components of Sigma
sig = Data_Train * Data_Train' / size(Data_Train, 2);
[U,S,V] = svd(sig);
% now reconstruct
new_train = U(:,1:d)' * Data_Train;
new_test = U(:,1:d)' * Data_Test;
% if you want to use raw data
% new_train=Data_Train;
% new_test=Dta_Test

% Training of SVM for different C values change: -C "Number"
% For linear kernel -t 0 for radial basis kernel t-2
model = svmtrain(Train_Labels, new_train', '-s 0 -t 0 -c 0.01');

% Now trained model can be used to classify.

% Classification for test data:
[predicted_label, accuracy, decision_values]=svmpredict(Test_Labels, new_test', model);

% Classification for training data:
[predicted_label_train, accuracy_train, decision_values_train]=svmpredict(Train_Labels, new_train', model);
fprintf('the test accuracy d=40 is %8.4f%%\n',accuracy(1));
fprintf('the train accuracy d=40 is %8.4f%%\n',accuracy_train(1));
