converter_train; %optional if didn't run already
load train_features_targets_2025.mat
train_features = features;
train_targets = targets;
clear features targets;

converter_test; %optional if didn't run already
load test_features_targets_2025.mat
test_features = features;
test_targets = targets;
clear features targets;