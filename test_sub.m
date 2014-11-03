clear, clc;
cd 'e:\documents and settings\ASALLAB\Desktop\Flash\PhD_Flash\Implementation\Diactrization\Results\Configuration_0.78';
load input_data mTestTargets mTestFeatures nBitfieldLength vChunkLength vOffset;

load final_net;

ctrCorrectlyClassified = 0;

cd 'e:\documents and settings\ASALLAB\Desktop\Flash\PhD_Flash\Implementation\Generic Classifier\Generic_Classifier_0.13';

for i = 1 : size(mTestFeatures, 1)
    [I1 vTargetOut]=max(mTestTargets(i,:), [], 2); % J1 is the index where max. output is found in the desired target
    
    [nErr, vTestTargetsOut] = TST_computeClassificationErrDNN(mTestFeatures(i,:), mTestTargets(i,:), NM_strNetParams, 0, 0, 0, 1, 1, 'ABSOLUTE_ERR_CALC', nBitfieldLength, vChunkLength, vOffset, 'Raw');
    
    ctrCorrectlyClassified = ctrCorrectlyClassified + (vTargetOut==vTestTargetsOut);
    
    fprintf(1, 'Accuracy = %d\n', ctrCorrectlyClassified / size(mTestFeatures, 1));
end