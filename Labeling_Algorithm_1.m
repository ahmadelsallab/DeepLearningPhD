% 1- Read data in data[Numcases, 728]
% - Load data[i].mat --> i: class number, i= 0:9
% Loop on i:

load mnist_weights;
curr_cluster_size = 0;
k = 10;
codes =[];
wr_idx = 1;
for i = 1 : k,
  %load digit data
  load(['digit' num2str(i-1) '.mat'],'-mat');

  %prev_cluster_size = curr_cluster_size;
  curr_cluster_size = size(D,1);
  labels(wr_idx : (wr_idx + curr_cluster_size - 1)) = (i)*ones(curr_cluster_size, 1);
  wr_idx = wr_idx + curr_cluster_size;  
  %Calculate code: Feed forward
  data = [D ones(curr_cluster_size,1)]; % ones: augmentation to account for biases. w1, w2,...w4 has the last row augmented with biases: w1=[vishid; hidrecbiases]; adding hidrecbiases as last row
  %data: curr_cluster_size X (numdims + 1) = curr_cluster_size X 729
  %w1: (numdims(728) + 1(bias)) X (numhid) = 729 X numhid
  w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(curr_cluster_size,1)]; 
  %w1probs: (curr_cluster_size) X (numhid(1000) + 1) = (curr_cluster_size) X 1001
  %w2: (numhid(1000) + 1) X numpen(500) = 1001 X 500
  w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(curr_cluster_size,1)];
  %w2probs: ((curr_cluster_size) X (numpen(500) + 1) = (curr_cluster_size) X 501
  %w3: (numpen (500) + 1(bias)) X (numpen2(250) = 501 X 250
  w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(curr_cluster_size,1)];
  %w3props: (curr_cluster_size) X 251
  %w4:(numpen2(250) + 1(bias)) X numopen = 251 X 30 
  code = w3probs*w4;
  %code: curr_cluster_size X 30
  
  %Calculate means
  means(i,:) = mean(code,1); %or mean(code) returns row contains average of every colomn
  
  % Append the codes
  % codes: totnumcases X 30 = sum(curr_cluster_size) X 30 
  codes = [codes; code];
  clear D;
  fprintf(1,'Digit = %d \n', i);
 end;
 
 % Apply K-means
 %[IDX,C] = kmeans(codes, 10, 'distance', 'sqEuclidean', 'onlinephase', 'off');
 [new_labels, new_means] = kmeans_dev_1(codes, k, labels, means);
 
%Calculate distances
for i = 1 : size(codes,1) % numcases
	for j = 1 : k %size(means,1) = k
		 %D(j,i) = euc_dist(codes(i,:), means(j,:));
		 D(j,i) = sum((codes(i,:) - means(j,:)).^2);
	end;
end;

 % Compare indicies and difference in means
 idx_err = size(find((new_labels-labels)== 0),2)/size(labels,2)*100 %sum((new_labels-label)s.^2)/size(labels, 1);
 [M N] = size(means);
 means_err = sum(sum((new_means-means).^2)) /(M*N);