clear, clc;
load test_features_targets_2025.mat;
err_ctr = 0;	
misclassified_index = [];
load mnistclassify_weights;
N=1;
numtargets=size(targets, 2);
confusion_matrix = zeros(numtargets, numtargets, 200);
confusion_matrix_freq = zeros(numtargets, numtargets);

for i = 1 : size(features,1)
	mail_num = i;
	classify_mail;%(i, features, targets);%, confusion_matrix, confusion_matrix_freq);
	if(result == 0)
		err_ctr = err_ctr + 1
		misclassified_index = [misclassified_index; i];
	end
end

err_ctr
misclassified_index