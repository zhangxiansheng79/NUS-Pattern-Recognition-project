Data_Train = loadMNISTImages('train-images.idx3-ubyte');
Train_Labels = loadMNISTLabels('train-labels.idx1-ubyte');
Data_Test = loadMNISTImages('t10k-images.idx3-ubyte');
Test_Labels = loadMNISTLabels('t10k-labels.idx1-ubyte');

% d is number of principal components 
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
reconstructed_train = U(:,1:d)' * Data_Train;
reconstructed_test = U(:,1:d)' * Data_Test;

% Neighrest neighbour classification I used function "nearestneighbour" 
% to find nearest neighbor 
for i=1:10000
    distance=sqrt(nearestneighbour(reconstructed_train,reconstructed_test(:,i)));
    [M, indexx] = min(distance);
    labels_for_test(i)=Train_Labels(indexx);
end

%This will give accuracy of test data
test_accuracy=1-(length(find(labels_for_test' ~= Test_Labels)))/length(Test_Labels);

% This will plot 3-D Distribution of Train Data with PCA
tencolors=fliplr(hsv(10));
for i=1:10
    ind=find(Train_Labels==i-1);
    plot3(reconstructed_train(1,ind),reconstructed_train(2,ind),reconstructed_train(3,ind),'v','color',tencolors(i,:))
    hold on;
end
legend('0','1','2','3','4','5','6','7','8','9')
xlabel('1st PC')
ylabel('2nd PC')
zlabel('3rd PC')
title('Projected Train Data in 3-D')


% This will plot 2-D Distribution of Train Data with PCA
figure
tencolors=fliplr(hsv(10));
for i=1:10
    ind=find(Train_Labels==i-1);
    plot(reconstructed_train(1,ind),reconstructed_train(2,ind),'v','color',tencolors(i,:))
    hold on;
end
legend('0','1','2','3','4','5','6','7','8','9')
xlabel('1st PC')
ylabel('2nd PC')
title('Projected Train Data in 2-D')


% This is for Visualization of eigenvectors of sample covariance matrix
% largest 40 eigenvectors:
figure
for i=1:40
    img = reshape( U(:,i), 28 , 28 );
    subplot( 5, 8, i );
    imshow( img, [] );
end


% d=154 will give %95 of energy of all eigenvalues
eigval=diag(S);
for nn=1:784
    optimal_va=(sum(eigval(1:nn)))/(sum(eigval(1:length(U))));
    if optimum_val>0.95
        fprintf('the number of pi is%d\n',nn);
        break;
    end
end