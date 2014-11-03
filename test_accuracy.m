clear, clc;
% Test data
load 'E:\Documents and Settings\ASALLAB\Desktop\Flash\PhD_Flash\Implementation\Diactrization\Results\Configuration_0.157\input_data' mTestTargets mTestFeatures nBitfieldLength vChunkLength vOffset;

% DNN final_net weigths
load 'e:\documents and settings\ASALLAB\Desktop\Flash\PhD_Flash\Implementation\Diactrization\Results\Configuration_0.144\final_net';

% Classifier path
cd 'e:\documents and settings\ASALLAB\Desktop\Flash\PhD_Flash\Implementation\Generic Classifier\Generic_Classifier_0.15';

ctrCorrectlyClassified = 0;
ctrTotal = 1;

% [I1 vTargets]=max(mTestTargets, [], 2); % J1 is the index where max. output is found in the desired target
% [nErr, vTargetsOut] = TST_computeClassificationErrDNN(mTestFeatures, mTestTargets, NM_strNetParams, 0, 0, 0, 1, 1, 'ABSOLUTE_ERR_CALC', nBitfieldLength, vChunkLength, vOffset, 'Raw');

% fprintf(1, 'Accuracy = %d\n', find(vTargets==vTargetsOut) / size(vTargets, 1));

for i = 1 : size(mTestFeatures, 1)
    [I1 vTargetOut]=max(mTestTargets(i,:), [], 2); % J1 is the index where max. output is found in the desired target
    
    [nErr, vTestTargetsOut] = TST_computeClassificationErrDNN(mTestFeatures(i,:), mTestTargets(i,:), NM_strNetParams, 0, 0, 0, 1, 1, 'ABSOLUTE_ERR_CALC', nBitfieldLength, vChunkLength, vOffset, 'Raw');

    ctrCorrectlyClassified = ctrCorrectlyClassified + (vTargetOut==vTestTargetsOut);  
    
    fprintf(1, 'Accuracy = %d\n', ctrCorrectlyClassified / ctrTotal);
    
    ctrTotal = ctrTotal + 1;
end
fprintf(1, 'Accuracy = %d\n', ctrCorrectlyClassified / ctrTotal);