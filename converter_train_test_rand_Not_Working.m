fid = fopen('..\Preprocessing\features_gadwal_binary.txt');%
%fid = fopen('features_binary_final_tmp.txt');

train_features=[]; 
train_targets=[]; 

test_features=[]; 
test_targets=[]; 

trainFactor = 3/4;
testFactor = 1/4;

mail_ctr = 0;
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

% Get train features
numTrainFeatures = trainFactor * size(features,1);
for b = 1 : numTrainFeatures
  train_features = features(randomorder(b), :);
  train_targets = targets(randomorder(b), :);
end;


% Get test features
numTestFeatures = testFactor * size(features,1);
for b = 1 : numTestFeatures
  test_features = features(randomorder(b), :);
  test_targets = targets(randomorder(b), :);
end;

fclose(fid);
save train_test_features_targets.mat test_features test_targets train_features train_targets;