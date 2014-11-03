clear, clc;
% Train deep auto-encoder - 2D%
wsddeepauto;

% Load 2-D code weights %
%load mnist_weights_2D w1 w2 w3 w4 w5 w6 w7 w8

N_dim = 2;

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
			plot(code(:,1), code(:,2), 'r+');
			%break;
		case 2
			plot(code(:,1), code(:,2), 'b+');
			%break;	
		case 3
			plot(code(:,1), code(:,2), 'g+');
			%break;
		case 4
			plot(code(:,1), code(:,2), 'y+');
			%break;
		case 5
			plot(code(:,1), code(:,2), 'c+');
			%break;            
		case 6
			plot(code(:,1), code(:,2), 'm+');
			%break;
		case 7
			plot(code(:,1), code(:,2), 'k+');
			%break;
		case 8
			plot(code(:,1), code(:,2), 'bs');
			%break;
		case 9
			plot(code(:,1), code(:,2), 'gs');
			%break;
		case 10
			plot(code(:,1), code(:,2), 'ys');
			%break;           
    end 
	hold on;

end
