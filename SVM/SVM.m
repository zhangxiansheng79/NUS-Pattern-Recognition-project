Data_Train = loadMNISTImages('train-images.idx3-ubyte');
Train_Labels = loadMNISTLabels('train-labels.idx1-ubyte');
Data_Test = loadMNISTImages('t10k-images.idx3-ubyte');
Test_Labels = loadMNISTLabels('t10k-labels.idx1-ubyte');

% Apply PCA to data comment it if you want to use raw data

% d is number of principal components you can change its number
d1=40;
d2=80;
d3=200;
% careful to remove mean of training data among all dimension from test 
% data, cannot use test data.

mean_data=mean(Data_Train,2);
Data_Train=Data_Train-repmat(mean_data,1,size(Data_Train,2));
Data_Test=Data_Test-repmat(mean_data,1,size(Data_Test,2));

% Find Principal Components of Sigma
sig = Data_Train * Data_Train' / size(Data_Train, 2);
[U,S,V] = svd(sig);
% now reconstruct
new_train1 = U(:,1:d1)' * Data_Train;
new_test1 = U(:,1:d1)' * Data_Test;
new_train2 = U(:,1:d2)' * Data_Train;
new_test2 = U(:,1:d2)' * Data_Test;
new_train3 = U(:,1:d3)' * Data_Train;
new_test3 = U(:,1:d3)' * Data_Test;
new_train4 = Data_Train;
new_test4 = Data_Test;


% Training of SVM for different C values change: -C "Number"
% For linear kernel -t 0 for radial basis kernel t-2
model1 = svmtrain(Train_Labels, new_train1', '-s 0 -t 0 -c 0.01');
model2 = svmtrain(Train_Labels, new_train1', '-s 0 -t 0 -c 0.1');
model3 = svmtrain(Train_Labels, new_train1', '-s 0 -t 0 -c 1');
model4 = svmtrain(Train_Labels, new_train1', '-s 0 -t 0 -c 10');

model5 = svmtrain(Train_Labels, new_train2', '-s 0 -t 0 -c 0.01');
model6 = svmtrain(Train_Labels, new_train2', '-s 0 -t 0 -c 0.1');
model7 = svmtrain(Train_Labels, new_train2', '-s 0 -t 0 -c 1');
model8 = svmtrain(Train_Labels, new_train2', '-s 0 -t 0 -c 10');

model9 = svmtrain(Train_Labels, new_train3', '-s 0 -t 0 -c 0.01');
model10 = svmtrain(Train_Labels, new_train3', '-s 0 -t 0 -c 0.1');
model11= svmtrain(Train_Labels, new_train3', '-s 0 -t 0 -c 1');
model12= svmtrain(Train_Labels, new_train3', '-s 0 -t 0 -c 10');

model13= svmtrain(Train_Labels, new_train4', '-s 0 -t 0 -c 0.01');
model14= svmtrain(Train_Labels, new_train4', '-s 0 -t 0 -c 0.1');
model15= svmtrain(Train_Labels, new_train4', '-s 0 -t 0 -c 1');
model16= svmtrain(Train_Labels, new_train4', '-s 0 -t 0 -c 10');
% Now trained model can be used to classify.

% Classification for test data and Classification for training data:
[predicted_label1, accuracy1, decision_values1]=svmpredict(Test_Labels, new_test1', model1);
[predicted_label2, accuracy2, decision_values2]=svmpredict(Test_Labels, new_test1', model2);
[predicted_label3, accuracy3, decision_values3]=svmpredict(Test_Labels, new_test1', model3);
[predicted_label4, accuracy4, decision_values4]=svmpredict(Test_Labels, new_test1', model4);
[predicted_label_train1, accuracy_train1, decision_values_train1]=svmpredict(Train_Labels, new_train1', model1);
[predicted_label_train2, accuracy_train2, decision_values_train2]=svmpredict(Train_Labels, new_train1', model2);
[predicted_label_train3, accuracy_train3, decision_values_train3]=svmpredict(Train_Labels, new_train1', model3);
[predicted_label_train4, accuracy_train4, decision_values_train4]=svmpredict(Train_Labels, new_train1', model4);
[predicted_label5, accuracy5, decision_values5]=svmpredict(Test_Labels, new_test2', model5);
[predicted_label6, accuracy6, decision_values6]=svmpredict(Test_Labels, new_test2', model6);
[predicted_label7, accuracy7, decision_values7]=svmpredict(Test_Labels, new_test2', model7);
[predicted_label8, accuracy8, decision_values8]=svmpredict(Test_Labels, new_test2', model8);
[predicted_label_train5, accuracy_train5, decision_values_train5]=svmpredict(Train_Labels, new_train2', model5);
[predicted_label_train6, accuracy_train6, decision_values_train6]=svmpredict(Train_Labels, new_train2', model6);
[predicted_label_train7, accuracy_train7, decision_values_train7]=svmpredict(Train_Labels, new_train2', model7);
[predicted_label_train8, accuracy_train8, decision_values_train8]=svmpredict(Train_Labels, new_train2', model8);
[predicted_label9,  accuracy9,  decision_values9]=svmpredict(Test_Labels, new_test3', model9);
[predicted_label10, accuracy10, decision_values10]=svmpredict(Test_Labels, new_test3', model10);
[predicted_label11, accuracy11, decision_values11]=svmpredict(Test_Labels, new_test3', model11);
[predicted_label12, accuracy12, decision_values12]=svmpredict(Test_Labels, new_test3', model12);
[predicted_label_train9, accuracy_train9, decision_values_train9]=svmpredict(Train_Labels, new_train3', model9);
[predicted_label_train10, accuracy_train10, decision_values_train10]=svmpredict(Train_Labels, new_train3', model10);
[predicted_label_train11, accuracy_train11, decision_values_train11]=svmpredict(Train_Labels, new_train3', model11);
[predicted_label_train12, accuracy_train12, decision_values_train12]=svmpredict(Train_Labels, new_train3', model12);
[predicted_label13,  accuracy13,  decision_values13]=svmpredict(Test_Labels, new_test4', model13);
[predicted_label14, accuracy14, decision_values14]=svmpredict(Test_Labels, new_test4', model14);
[predicted_label15, accuracy15, decision_values15]=svmpredict(Test_Labels, new_test4', model15);
[predicted_label16, accuracy16, decision_values16]=svmpredict(Test_Labels, new_test4', model16);
[predicted_label_train13, accuracy_train13, decision_values_train13]=svmpredict(Train_Labels, new_train4', model13);
[predicted_label_train14, accuracy_train14, decision_values_train14]=svmpredict(Train_Labels, new_train4', model14);
[predicted_label_train15, accuracy_train15, decision_values_train15]=svmpredict(Train_Labels, new_train4', model15);
[predicted_label_train16, accuracy_train16, decision_values_train16]=svmpredict(Train_Labels, new_train4', model16);


fprintf('the test accuracy for d=40,c=0.01 is %8.4f\n',accuracy1(1));
fprintf('the train accuracy for d=40,c=0.01 is %8.4f\n',accuracy_train1(1));
fprintf('the test accuracy for d=40,c=0.1 is %8.4f\n',accuracy2(1));
fprintf('the train accuracy for d=40,c=0.1 is %8.4f\n',accuracy_train2(1));
fprintf('the test accuracy for d=40,c=1 is %8.4f\n',accuracy3(1));
fprintf('the train accuracy for d=40,c=1 is %8.4f\n',accuracy_train3(1));
fprintf('the test accuracy for d=40,c=10 is %8.4f\n',accuracy4(1));
fprintf('the train accuracy for d=40,c=10 is %8.4f\n',accuracy_train4(1));

fprintf('the test accuracy for d=80,c=0.01 is %8.4f\n',accuracy5(1));
fprintf('the train accuracy for d=80,c=0.01 is %8.4f\n',accuracy_train5(1));
fprintf('the test accuracy for d=80,c=0.1 is %8.4f\n',accuracy6(1));
fprintf('the train accuracy for d=80,c=0.1 is %8.4f\n',accuracy_train6(1));
fprintf('the test accuracy for d=80,c=1 is %8.4f\n',accuracy7(1));
fprintf('the train accuracy for d=80,c=1 is %8.4f\n',accuracy_train7(1));
fprintf('the test accuracy for d=80,c=10 is %8.4f\n',accuracy8(1));
fprintf('the train accuracy for d=80,c=10 is %8.4f\n',accuracy_train8(1));

fprintf('the test accuracy for d=200,c=0.01 is %8.4f\n',accuracy9(1));
fprintf('the train accuracy for d=200,c=0.01 is %8.4f\n',accuracy_train9(1));
fprintf('the test accuracy for d=200,c=0.1 is %8.4f\n',accuracy10(1));
fprintf('the train accuracy for d=200,c=0.1 is %8.4f\n',accuracy_train10(1));
fprintf('the test accuracy for d=200,c=1 is %8.4f\n',accuracy11(1));
fprintf('the train accuracy for d=200,c=1 is %8.4f\n',accuracy_train11(1));
fprintf('the test accuracy for d=200,c=10 is %8.4f\n',accuracy12(1));
fprintf('the train accuracy for d=200,c=10 is %8.4f\n',accuracy_train12(1));

fprintf('the test accuracy for raw data,c=0.01 is %8.4f\n',accuracy13(1));
fprintf('the train accuracy for raw data,c=0.01 is %8.4f\n',accuracy_train13(1));
fprintf('the test accuracy for raw data,c=0.1 is %8.4f\n',accuracy14(1));
fprintf('the train accuracy for raw data,c=0.1 is %8.4f\n',accuracy_train14(1));
fprintf('the test accuracy for raw data,c=1 is %8.4f\n',accuracy15(1));
fprintf('the train accuracy for raw data,c=1 is %8.4f\n',accuracy_train15(1));
fprintf('the test accuracy for raw data,c=10 is %8.4f\n',accuracy16(1));
fprintf('the train accuracy for raw data,c=10 is %8.4f\n',accuracy_train16(1));






