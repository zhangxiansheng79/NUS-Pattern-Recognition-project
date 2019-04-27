Data_Train = loadMNISTImages('train-images.idx3-ubyte');
Train_Labels = loadMNISTLabels('train-labels.idx1-ubyte');
Data_Test = loadMNISTImages('t10k-images.idx3-ubyte');
Test_Labels = loadMNISTLabels('t10k-labels.idx1-ubyte');

% q is PCA dimension reduction 
% d is LDA dimension reduction
q=80;
d1=2;
d2=3;
d3=9;


% Use PCA to reduce dimension first before LDA.
% remove mean of training data among all dimension from test 
% data
mean_data=mean(Data_Train,2);
Data_Train=Data_Train-repmat(mean_data,1,size(Data_Train,2));
Data_Test=Data_Test-repmat(mean_data,1,size(Data_Test,2));

% Find Principal Components of Sigma
sig = Data_Train * Data_Train' / size(Data_Train, 2);
[U,S,V] = svd(sig);

% now reconstruct with PCA
new_train = U(:,1:q)' * Data_Train;
new_test = U(:,1:q)' * Data_Test;

% After PCA implementation? now LDA can be implemented
receivemean_retr = mean(new_train, 2); % mean of each reconstructed image
class_mean = zeros(q, 10);
num_of_class = zeros(1, 10);
for i = 1:10
    class_mean(:, i) = mean(new_train(:,(Train_Labels == i-1)), 2);
    num_of_class(i) = size(new_train(:,(Train_Labels == i-1)), 2);
end

% Compute Sw and Sb
S_w = zeros(q, q);
S_b = zeros(q, q);
% This propability of each class
class_prob=num_of_class./sum(num_of_class);

% Compute Scatter Matrix
for i = 1:10 % For all classes
    % Within Class Scatter
    index=find(Train_Labels == i-1);
    S_i = zeros(q, q);
    for j = 1:length(index)
        x = new_train(:, index(j));
        S_i = S_i + (x - class_mean(:,i)) * (x - class_mean(:,i))';
    end
    S_i=S_i*(1/num_of_class(i));
    S_w = S_w + S_i*class_prob(i);
    % Between Class Scatter
    S_b = S_b + (num_of_class(i)/size(new_train, 2)) * (class_mean(:, i) - receivemean_retr) * (class_mean(:, i) - receivemean_retr)';
end

% Eigenvalue Decomposition matrix inv(Sw)*Sb
[eigvector,eigvalue] = eig(S_b,S_w);
eigs=diag(eigvalue);
% Store indicies from max to min
[c, index] = sort(eigs,'descend'); 
EigenVectors1=eigvector(:,index(1:d1));
EigenVectors2=eigvector(:,index(1:d2));
EigenVectors3=eigvector(:,index(1:d3));
EigenVectors=eigvector(:,index(1:9));


% Reduce data dimension to d (LDA) and reconstruct
new_train1 = EigenVectors1'*new_train;
new_test1 = EigenVectors1'*new_test;
new_train2 = EigenVectors2'*new_train;
new_test2 = EigenVectors2'*new_test;
new_train3 = EigenVectors3'*new_train;
new_test3 = EigenVectors3'*new_test;
new_train4=EigenVectors'*new_train;
new_test4=EigenVectors'*new_test;
% Neighrest neighbour classification I used function "nearestneighbour" 
% to find nearest neighbor
for i=1:10000
    distance1=sqrt(nearestneighbour(new_train1,new_test1(:,i)));
    [M1, indexx1] = min(distance1);
    labels_for_test1(i)=Train_Labels(indexx1);
    distance2=sqrt(nearestneighbour(new_train2,new_test2(:,i)));
    [M2, indexx2] = min(distance2);
    labels_for_test2(i)=Train_Labels(indexx2);
    distance3=sqrt(nearestneighbour(new_train3,new_test3(:,i)));
    [M3, indexx3] = min(distance3);
    labels_for_test3(i)=Train_Labels(indexx3);

end

%This will give accuracy of test data
test_accuracy1=1-(length(find(labels_for_test1' ~= Test_Labels)))/length(Test_Labels);
test_accuracy2=1-(length(find(labels_for_test2' ~= Test_Labels)))/length(Test_Labels);
test_accuracy3=1-(length(find(labels_for_test3' ~= Test_Labels)))/length(Test_Labels);

fprintf('the test accuracy of reducing data dimensionality from 784 to 2 is %8.4f\n',test_accuracy1);
fprintf('the test accuracy of reducing data dimensionality from 784 to 3 is %8.4f\n',test_accuracy2);
fprintf('the test accuracy of reducing data dimensionality from 784 to 9 is %8.4f\n',test_accuracy3);

% This will plot 3-D Distribution of Train Data with LDA
tencolors=fliplr(hsv(10));
for i=1:10
    ind=find(Train_Labels==i-1);
    plot3(new_train4(1,ind),new_train4(2,ind),new_train4(3,ind),'v','color',tencolors(i,:))
    hold on;
end
legend('0','1','2','3','4','5','6','7','8','9')
xlabel('1st PC')
ylabel('2nd PC')
zlabel('3rd PC')
title('Projected Train Data in 3-D for LDA')


% This will plot 2-D Distribution of Train Data with LDA
tencolors=fliplr(hsv(10));
figure
for i=1:10
    ind=find(Train_Labels==i-1);
    plot(new_train4(1,ind),new_train4(2,ind),'v','color',tencolors(i,:))
    hold on;
end
legend('0','1','2','3','4','5','6','7','8','9')
xlabel('1st PC')
ylabel('2nd PC')
title('Projected Train Data in 2-D for LDA')
