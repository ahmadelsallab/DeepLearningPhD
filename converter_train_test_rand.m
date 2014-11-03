fid = fopen('..\Preprocessing\features_gadwal_binary.txt');%
%fid = fopen('features_binary_final_tmp.txt');

train_features=[]; 
train_targets=[]; 

test_features=[]; 
test_targets=[]; 

train_factor = 8;

% Read all features and targets
s = fgets(fid);
features = [];
targets = [];
target_feature = 1;
while(s > 0)
	if target_feature == 0
		features = [features; str2num(s)];
	else
		targets = [targets; str2num(s)];
	end
	target_feature = ~target_feature;
	s = fgets(fid);
end;

% Randomize
rand('state',0); %so we know the permutation of the training data
randomorder=randperm(size(features,1));

for b = 1 : size(features,1)
	rand_features = features(randomorder(b), :);
	rand_targets = targets(randomorder(b), :);
	if (mod(b-1, train_factor) == 0)	
		test_targets = [test_targets; rand_targets];
		test_features = [test_features; rand_features];
	else
		train_targets = [train_targets; rand_targets];
		train_features = [train_features; rand_features];
	end;
end;
fclose(fid);
mail_ctr = 0;




save train_test_features_targets.mat test_features test_targets train_features train_targets;