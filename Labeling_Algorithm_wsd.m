% Load auto encoder weights
load mnist_weights;

% Load train features and desired labels
load train_test_features_targets.mat train_features train_targets;

code_dim = size(w4, 2);

numtargets = size(train_targets, 2);

numfeatures = size(train_features, 1);

% Initialize codes
codes = [];
  
% Initialize clusters means
means = zeros(numtargets, code_dim);

% Loop on all train features
numfeatures_perlabel = zeros(numtargets, 1);
for i = 1 : numfeatures
  
  % The actual of the feature is the position that has 1 in the desired target
  labels(i) = find(train_targets(i,:)==1);
  
  %Calculate code: Feed forward
  data = [train_features(i, :) 1]; % ones: augmentation to account for biases. w1, w2,...w4 has the last row augmented with biases: w1=[vishid; hidrecbiases]; adding hidrecbiases as last row
  %data: 1 X (numfeatures + 1) 
  %w1: (numdims + 1(bias)) X (numhid)
  w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  1]; 
  %w1probs: (1 X (numhid(1000) + 1) = 1 X 1001
  %w2: (numhid(1000) + 1) X numpen(500) = 1001 X 500
  w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs 1];
  %w2probs: ((11) X (numpen(500) + 1) = (c1) X 501
  %w3: (numpen (500) + 1(bias)) X (numpen2(250) = 501 X 250
  w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  1];
  %w3props: (1) X 251
  %w4:(numpen2(250) + 1(bias)) X numopen = 251 X 30 
  code = w3probs*w4;
  %code: 1 X 30
  
  %Append to the means of the label of the current feature
  means(labels(i),:) = means(labels(i),:) + code; %or mean(code) returns row contains average of every colomn
  numfeatures_perlabel(labels(i)) = numfeatures_perlabel(labels(i)) + 1;
  % Append the codes
  % codes: totnumcases X 30 = sum(curr_cluster_size) X 30 
  codes = [codes; code];
  
  fprintf(1,'training example = %d \n', i);
 end
 
%Calculate means of each cluster to inintialize k-means algorithm 
for i = 1 : numtargets
	means(i,:) = means(i,:)/numfeatures_perlabel(i);
end
 
 % Apply K-means
 %[IDX,C] = kmeans(codes, 10, 'distance', 'sqEuclidean', 'onlinephase', 'off');
 [new_labels, new_means] = kmeans_dev_1(codes, numtargets, labels, means);
 
%Calculate distances
for i = 1 : size(codes,1) % numcases
	for j = 1 : numtargets
		 %D(j,i) = euc_dist(codes(i,:), means(j,:));
		 D(j,i) = sum((codes(i,:) - means(j,:)).^2);
	end;
end;

 % Compare indicies and difference in means
 idx_accuracy = size(find((new_labels-labels)== 0),2)/size(labels,2)*100 %sum((new_labels-label)s.^2)/size(labels, 1);
 [M N] = size(means);
 means_err = sum(sum((new_means-means).^2)) /(M*N);
 
 save auto_label.mat;