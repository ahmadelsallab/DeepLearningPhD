clear, clc;
% Train deep auto-encoder - 3D%
enrondeepauto;

% Load 3-D code weights %
%load mnist_weights_3D w1 w2 w3 w4 w5 w6 w7 w8
%load mnist_weights w1 w2 w3 w4 w5 w6 w7 w8

N_dim = 3;

data = [];

load '..\Important MATLAB workspaces\arch_500_500_2000_1000_features_10_targets_kitchen_l' numtargets;

load train_test_features_targets.mat test_features test_targets train_features train_targets;

for i = 1 : numtargets
	fprintf(1, 'Obtaining data of class %d\n', i);
    clear data;
	data = [];
	for j = 1 : size(train_features, 1)
		if(i == find(train_targets(j,:)==1))
			data = [data; train_features(j,:)];
		end
    end
    
    fprintf(1, 'Obtaining codes of class %d\n', i);
    
	N = size(data, 1);
	code = zeros(N,N_dim);
    data = [data ones(N,1)];
	w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)];
	w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
	w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
	code = w3probs*w4;
	
    fprintf(1, 'Plot codes of class %d\n', i);
	switch (i)
		case 1
			plot3(code(:,1), code(:,2), code(:,3), 'r+');
			%break;
		case 2
			plot3(code(:,1), code(:,2), code(:,3), 'b+');
			%break;	
		case 3
			plot3(code(:,1), code(:,2), code(:,3), 'g+');
			%break;
		case 4
			plot3(code(:,1), code(:,2), code(:,3), 'y+');
			%break;
	end 
	hold on;

end
