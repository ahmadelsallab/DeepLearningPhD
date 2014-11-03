% Version 1.000
%
% Code provided by Ruslan Salakhutdinov and Geoff Hinton
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.
%load_train_test;
maxrange = 1;
if CONFIG_bRandFeatures == 1
	converter_train_test_rand;
else
	converter_train_test;
end
load train_test_features_targets.mat test_features test_targets train_features train_targets;

load auto_label.mat new_labels numtargets;

% Mark targets according to auto labels
for i = 1 : size(new_labels,2)
	
	for j = 1 : numtargets
		if(new_labels(i)==j)
			train_targets(i,j) = 1;
		else
			train_targets(i,j) = 0;
		end
	end
end

% Training data
train_features = train_features/maxrange;

totnum=size(train_features,1);
fprintf(1, 'Size of the training dataset= %5d \n', totnum);

rand('state',0); %so we know the permutation of the training data
randomorder=randperm(totnum);


numdims  =  size(test_features,2);
batchsize = 10;
numbatches=floor(totnum/batchsize);
batchdata = zeros(batchsize, numdims, numbatches);
numtargets = size(train_targets, 2);
batchtargets = zeros(batchsize, numtargets, numbatches);

for b=1:numbatches
  batchdata(:,:,b) = train_features(randomorder(1+(b-1)*batchsize:b*batchsize), :);
  batchtargets(:,:,b) = train_targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end;
clear train_features train_targets;

% Test data
test_features = test_features/maxrange;

totnum=size(test_features,1);
fprintf(1, 'Size of the test dataset= %5d \n', totnum);

rand('state',0); %so we know the permutation of the training data
randomorder=randperm(totnum);


numdims  =  size(test_features,2);
batchsize = 10;
numbatches=floor(totnum/batchsize);
testbatchdata = zeros(batchsize, numdims, numbatches);
testbatchtargets = zeros(batchsize, numtargets, numbatches);

for b=1:numbatches
  testbatchdata(:,:,b) = test_features(randomorder(1+(b-1)*batchsize:b*batchsize), :);
  testbatchtargets(:,:,b) = test_targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end;
clear test_features test_targets;


%%% Reset random seeds 
rand('state',sum(100*clock)); 
randn('state',sum(100*clock)); 



