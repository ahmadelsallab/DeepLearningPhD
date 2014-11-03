% new_labels, init_labels: N_cases X 1
% new_means, init_means: k X N_dim
% k: 1x1
% codes: N_cases X N_dim
function [new_labels, new_means] = kmeans_dev_1(codes, k, init_labels, init_means)
% Initialize labels and means
means = init_means;
labels = init_labels;
old_means = init_means;
old_labels = init_labels;
count = 0;
err = 100;
    while(err > 0)

        count = count + 1;

        %Calculate distances
        for i = 1 : size(codes,1) % numcases
            for j = 1 : k %size(means,1) = k
                %D(j,i) = euc_dist(codes(i,:), means(j,:));
                D(j,i) = sum((codes(i,:) - means(j,:)).^2);
            end;
        end;

        %Calculate new labels
        [min_dist , labels] = min(D,[],1);

        %Calculate new means
        for i = 1 : k
            ind = [];
            ind = find(labels == i);
            sum_codes = zeros(1, size(codes,2));	
            for j = 1 : size(ind, 2)
                sum_codes = sum_codes + codes(ind(j),:);
            end;
            if(size(ind, 2) == 0)
               means(i,:) = sum_codes./0.0001; 
            else
                means(i,:) = sum_codes./size(ind, 2);
            end
            clear ind;
        end;

        % Compute error
        [M N] = size(means);
        err = sum(sum((means - old_means).^2)) /(M*N);

        % Update means and labels
        idx_err = size(find((labels-old_labels)== 0),2)/size(labels,2)*100;
        old_means = means;
        old_labels = labels;
         
        fprintf(1,'Round %d Error = %d  Labels error = %d \n', count, err, idx_err);
    end;
	
	%Calculate distances
	for i = 1 : size(codes,1) % numcases
			for j = 1 : k %size(means,1) = k
				 %D(j,i) = euc_dist(codes(i,:), means(j,:));
				 D(j,i) = sum((codes(i,:) - means(j,:)).^2);
			end;
	end;
	
	%Calculate mean examples indices = indices of the nearest examples to the final means
	% Note that: no need to recaluclate distances again since the err is zero between old and new means in the final iteration, and hence the distances D[j,i] are the same
  [min_dist , mean_examples_indices] = min(D,[],2);
  
  
  % Apply original label of the mean example to the whole cluster
  % 1. Get correct label (the label of the nearest example to the mean
%	[C1 C2 C3]mean_examples_indices = nearest to the mean Li
%L1				|C2 idx=1
%L2				|C1 idx=0
%L3				|C3 idx=2
%--------------------	
%labels: L2 L3 L1
%correct_labels = init_labels(1) init_labels(0) init_labels(2)

  for j = 1 : k
	correct_label(j) = init_labels(mean_examples_indices(j)); % correct_label = initial label of the example at the index nearest to the mean
  end;
  
  %2. Correct the labels
  new_labels = labels;
  for i = 1 : k
		ind = [];
		ind = find(labels == i);
		for j = 1 : size(ind,2)
			new_labels(ind(j)) = correct_label(i);
		end;
		clear ind;
  end;

new_means = means;
