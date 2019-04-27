Data_Train = loadMNISTImages('train-images.idx3-ubyte');
Train_Labels = loadMNISTLabels('train-labels.idx1-ubyte');
Data_Test = loadMNISTImages('t10k-images.idx3-ubyte');
Test_Labels = loadMNISTLabels('t10k-labels.idx1-ubyte');

% di is number of principal components 
d=40;
d1=40;
d2=80;
d3=200;
% remove mean of training data among all dimension from test 
% data

mean_data=mean(Data_Train,2);
Data_Train=Data_Train-repmat(mean_data,1,size(Data_Train,2));
Data_Test=Data_Test-repmat(mean_data,1,size(Data_Test,2));

% Find Principal Components of Sigma
sig = Data_Train * Data_Train' / size(Data_Train, 2);
[U,S,V] = svd(sig);

% now reconstruct
reconstructed_train = U(:,1:d)' * Data_Train;
reconstructed_test = U(:,1:d)' * Data_Test;
reconstructed_train1 = U(:,1:d1)' * Data_Train;
reconstructed_test1 = U(:,1:d1)' * Data_Test;
reconstructed_train2 = U(:,1:d2)' * Data_Train;
reconstructed_test2 = U(:,1:d2)' * Data_Test;
reconstructed_train3 = U(:,1:d3)' * Data_Train;
reconstructed_test3 = U(:,1:d3)' * Data_Test;

% for the Neighrest neighbour classification ,use function "nearestneighbour"
% in the nearestneighbour.m
for i=1:10000
    distance1=sqrt(nearestneighbour(reconstructed_train1,reconstructed_test1(:,i)));
    [M1, indexx1] = min(distance1);
    labels_for_test1(i)=Train_Labels(indexx1);
    distance2=sqrt(nearestneighbour(reconstructed_train2,reconstructed_test2(:,i)));
    [M2, indexx2] = min(distance2);
    labels_for_test2(i)=Train_Labels(indexx2);  
    distance3=sqrt(nearestneighbour(reconstructed_train3,reconstructed_test3(:,i)));
    [M3, indexx3] = min(distance3);
    labels_for_test3(i)=Train_Labels(indexx3);
    
end

%accuracy of test data
test_accuracy1=1-(length(find(labels_for_test1' ~= Test_Labels)))/length(Test_Labels);
test_accuracy2=1-(length(find(labels_for_test2' ~= Test_Labels)))/length(Test_Labels);
test_accuracy3=1-(length(find(labels_for_test3' ~= Test_Labels)))/length(Test_Labels);
fprintf('the test accuracy d=40 is %8.4f\n',test_accuracy1);
fprintf('the test accuracy d=80 is %8.4f\n',test_accuracy2);
fprintf('the test accuracy d=200 is %8.4f\n',test_accuracy3);
%3D Distribution of Train Data with PCA
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


%2D Distribution of Train Data with PCA
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


%Visualization of eigenvectors of sample covariance matrix
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
    optimal_value=(sum(eigval(1:nn)))/(sum(eigval(1:length(U))));
    if optimal_value>0.95
        fprintf('the number of pi is%d\n',nn);
        break;
    end
end
% compute the accuracy of d which is %95 of energy of all eigenvalues
d4=154;
new_train4 = U(:,1:d4)' * Data_Train;
new_test4 = U(:,1:d4)' * Data_Test;
for i=1:10000
    distance4=sqrt(nearestneighbour(new_train4,new_test4(:,i)));
    [M4, indexx4] = min(distance4);
    labels_for_test4(i)=Train_Labels(indexx4);
end
test_accuracy4=1-(length(find(labels_for_test4' ~= Test_Labels)))/length(Test_Labels);
fprintf('the test accuracy d=154 is %8.4f\n',test_accuracy4);
