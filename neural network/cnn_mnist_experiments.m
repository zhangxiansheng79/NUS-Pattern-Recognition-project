% Experiment with the cnn_mnist_fc_bnorm

[net_bn, info_bn] = cnn_mnist(...
  'expDir', 'data/mnist-bnorm', 'batchNormalization', true);

[net_fc, info_fc] = cnn_mnist(...
  'expDir', 'data/mnist-baseline', 'batchNormalization', false);

figure(1) ; clf ;
subplot(1,2,1) ;
a1=[info_fc.train.objective]';
a2=[info_fc.val.objective]';
a3=[info_bn.train.objective]';
a4=[info_bn.val.objective]';
plot([info_fc.train.objective]', 'o-') ; hold all ;
plot([info_fc.val.objective]', '*-') ;
plot([info_bn.train.objective]', '+--') ;
plot([info_bn.val.objective]', 'x--') ;
xlabel('epoch times'); ylabel('error') ;
grid on ;
h=legend('BSLN-train','BSLN-test','BNORM-train','BNORM-test') ;
set(h,'color','none');
title('loss') ;
subplot(1,2,2) ;
b1=1-[info_fc.train.top1err]';
b2=1-[info_fc.val.top1err]';
b3=1-[info_bn.train.top1err]';
b4=1-[info_bn.val.top1err]';
plot(1-[info_fc.train.top1err]', 'o-') ; hold all ;
plot(1-[info_fc.val.top1err]', '*-') ;
plot(1-[info_bn.train.top1err]', '+--') ;
plot(1-[info_bn.val.top1err]', 'x--') ;
h=legend('BSLN-train','BSLN-test','BNORM-train','BNORM-test') ;
grid on ;
xlabel('epoch times'); ylabel('accuracy') ;
set(h,'color','none') ;
title('accuracy') ;
drawnow ;
fprintf('the train accuracy of baseline data after 1 epoch is %8.4f%%\n',100*b1(1));
fprintf('loss is %8.4f\n',a1(1));
fprintf('the test accuracy of baseline data after 1 epoch is %8.4f%%\n',100*b2(1));
fprintf('loss is %8.4f\n',a2(1));
fprintf('the train accuracy of baseline data after 30 epoches is %8.4f%%\n',100*b1(20));
fprintf('loss is %8.4f\n',a1(20));
fprintf('the test accuracy of baseline data after 30 epoches is %8.4f%%\n',100*b2(20));
fprintf('loss is %8.4f\n',a2(20));

fprintf('the train accuracy of bnorm data after 1 epoch is %8.4f%%\n',100*b3(1));
fprintf('loss is %8.4f\n',a3(1));
fprintf('the test accuracy of bnorm data after 1 epoch is %8.4f%%\n',100*b4(1));
fprintf('loss is %8.4f\n',a4(1));
fprintf('the train accuracy of bnorm data after 30 epoches is %8.4f%%\n',100*b3(20));
fprintf('loss is %8.4f\n',a3(20));
fprintf('the test accuracy of bnorm data after 30 epoches is %8.4f%%\n',100*b4(20));
fprintf('loss is %8.4f\n',a4(20));


