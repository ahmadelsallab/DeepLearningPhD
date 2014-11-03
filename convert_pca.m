load input_data mTestFeatures;
[COEFF,mTestFeatures,latent] = princomp(mTestFeatures);
mTestFeatures = mTestFeatures(:,1:90);
load input_data_ mTrainFeatures;
[COEFF,mTrainFeatures,latent] = princomp(mTrainFeatures);
mTrainFeatures = mTrainFeatures(:,1:90);
load input_data_ mTrainTargets mTestTargets nBitfieldLength vChunkLength vOffset
save input_data
clear
clc