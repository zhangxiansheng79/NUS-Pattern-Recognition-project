Data_Train = loadMNISTImages('train-images.idx3-ubyte');
Train_Labels = loadMNISTLabels('train-labels.idx1-ubyte');
Data_Test = loadMNISTImages('t10k-images.idx3-ubyte');
Test_Labels = loadMNISTLabels('t10k-labels.idx1-ubyte');

% Apply PCA to data comment it if you want to use raw data

% d is number of principal components you can change its number
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

% For raw data uncomment next 2 lines
% new_train = Data_Train;
% new_test = Data_Test;


% Training of SVM for different C values change: -C "Number"
% For linear kernel -t 0 for radial basis kernel t-2
model1 = svmtrain(Train_Labels, new_train', '-s 0 -t 2 -c 0.01 -g 0.1');
model2 = svmtrain(Train_Labels, new_train', '-s 0 -t 2 -c 0.01 -g 1');
model3 = svmtrain(Train_Labels, new_train', '-s 0 -t 2 -c 0.01 -g 10');
model4 = svmtrain(Train_Labels, new_train', '-s 0 -t 2 -c 0.1 -g 0.1');
model5 = svmtrain(Train_Labels, new_train', '-s 0 -t 2 -c 0.1 -g 1');
model6 = svmtrain(Train_Labels, new_train', '-s 0 -t 2 -c 0.1 -g 10');
model7 = svmtrain(Train_Labels, new_train', '-s 0 -t 2 -c 1 -g 0.1');
model8 = svmtrain(Train_Labels, new_train', '-s 0 -t 2 -c 1 -g 1');
model9 = svmtrain(Train_Labels, new_train', '-s 0 -t 2 -c 1 -g 10');
model10 = svmtrain(Train_Labels, new_train', '-s 0 -t 2 -c 10 -g 0.1');
model11 = svmtrain(Train_Labels, new_train', '-s 0 -t 2 -c 10 -g 1');
model12 = svmtrain(Train_Labels, new_train', '-s 0 -t 2 -c 10 -g 10');
% Now trained model can be used to classify.

% Classification for test data:
[predicted_label1, accuracy1, decision_values1]=svmpredict(Test_Labels, new_test', model1);
[predicted_label2, accuracy2, decision_values2]=svmpredict(Test_Labels, new_test', model2);
[predicted_label3, accuracy3, decision_values3]=svmpredict(Test_Labels, new_test', model3);
[predicted_label4, accuracy4, decision_values4]=svmpredict(Test_Labels, new_test', model4);
[predicted_label5, accuracy5, decision_values5]=svmpredict(Test_Labels, new_test', model5);
[predicted_label6, accuracy6, decision_values6]=svmpredict(Test_Labels, new_test', model6);
[predicted_label7, accuracy7, decision_values7]=svmpredict(Test_Labels, new_test', model7);
[predicted_label8, accuracy8, decision_values8]=svmpredict(Test_Labels, new_test', model8);
[predicted_label9, accuracy9, decision_values9]=svmpredict(Test_Labels, new_test', model9);
[predicted_label10, accuracy10, decision_values10]=svmpredict(Test_Labels, new_test', model10);
[predicted_label11, accuracy11, decision_values11]=svmpredict(Test_Labels, new_test', model11);
[predicted_label12, accuracy12, decision_values12]=svmpredict(Test_Labels, new_test', model12);
% Classification for training data:
[predicted_label_train1, accuracy_train1, decision_values_train1]=svmpredict(Train_Labels, new_train', model1);
[predicted_label_train2, accuracy_train2, decision_values_train2]=svmpredict(Train_Labels, new_train', model2);
[predicted_label_train3, accuracy_train3, decision_values_train3]=svmpredict(Train_Labels, new_train', model3);
[predicted_label_train4, accuracy_train4, decision_values_train4]=svmpredict(Train_Labels, new_train', model4);
[predicted_label_train5, accuracy_train5, decision_values_train5]=svmpredict(Train_Labels, new_train', model5);
[predicted_label_train6, accuracy_train6, decision_values_train6]=svmpredict(Train_Labels, new_train', model6);
[predicted_label_train7, accuracy_train7, decision_values_train7]=svmpredict(Train_Labels, new_train', model7);
[predicted_label_train8, accuracy_train8, decision_values_train8]=svmpredict(Train_Labels, new_train', model8);
[predicted_label_train9, accuracy_train9, decision_values_train9]=svmpredict(Train_Labels, new_train', model9);
[predicted_label_train10, accuracy_train10, decision_values_train10]=svmpredict(Train_Labels, new_train', model10);
[predicted_label_train11, accuracy_train11, decision_values_train11]=svmpredict(Train_Labels, new_train', model11);
[predicted_label_train12, accuracy_train12, decision_values_train12]=svmpredict(Train_Labels, new_train', model12);

fprintf('d=40,the test accuracy for c=0.01,gamma=0.1 is %8.4f\n',accuracy1(1));
fprintf('d=40,the train accuracy for c=0.01,gamma=0.1 is %8.4f\n',accuracy_train1(1));
fprintf('d=40,the test accuracy for c=0.01,gamma=1 is %8.4f\n',accuracy2(1));
fprintf('d=40,the train accuracy for c=0.01,gamma=1 is %8.4f\n',accuracy_train2(1));
fprintf('d=40,the test accuracy for c=0.01,gamma=10 is %8.4f\n',accuracy3(1));
fprintf('d=40,the train accuracy for c=0.01,gamma=10 is %8.4f\n',accuracy_train3(1));

fprintf('d=40,the test accuracy for c=0.1,gamma=0.1 is %8.4f\n',accuracy4(1));
fprintf('d=40,the train accuracy for c=0.1,gamma=0.1 is %8.4f\n',accuracy_train4(1));
fprintf('d=40,the test accuracy for c=0.1,gamma=1 is %8.4f\n',accuracy5(1));
fprintf('d=40,the train accuracy for c=0.1,gamma=1 is %8.4f\n',accuracy_train5(1));
fprintf('d=40,the test accuracy for c=0.1,gamma=10 is %8.4f\n',accuracy6(1));
fprintf('d=40,the train accuracy for c=0.1,gamma=10 is %8.4f\n',accuracy_train6(1));

fprintf('d=40,the test accuracy for c=1,gamma=0.1 is %8.4f\n',accuracy7(1));
fprintf('d=40,the train accuracy for c=1,gamma=0.1 is %8.4f\n',accuracy_train7(1));
fprintf('d=40,the test accuracy for c=1,gamma=1 is %8.4f\n',accuracy8(1));
fprintf('d=40,the train accuracy for c=1,gamma=1 is %8.4f\n',accuracy_train8(1));
fprintf('d=40,the test accuracy for c=1,gamma=10 is %8.4f\n',accuracy9(1));
fprintf('d=40,the train accuracy for c=1,gamma=10 is %8.4f\n',accuracy_train9(1));

fprintf('d=40,the test accuracy for c=10,gamma=0.1 is %8.4f\n',accuracy10(1));
fprintf('d=40,the train accuracy for c=10,gamma=0.1 is %8.4f\n',accuracy_train10(1));
fprintf('d=40,the test accuracy for c=10,gamma=1 is %8.4f\n',accuracy11(1));
fprintf('d=40,the train accuracy for c=10,gamma=1 is %8.4f\n',accuracy_train11(1));
fprintf('d=40,the test accuracy for c=10,gamma=10 is %8.4f\n',accuracy12(1));
fprintf('d=40,the train accuracy for c=10,gamma=10 is %8.4f\n',accuracy_train12(1));
