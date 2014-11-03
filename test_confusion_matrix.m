clear, clc;
load input_data;
load final_net;
[CONFIG_strParams] = CONFIG_setConfigParams();
nBatchSize = 1000;
[mTestBatchData, mTestBatchTargets] = BM_randomizeBatchData(nBatchSize, mTestFeatures, mTestTargets);

%[nErr, vTargetsDNN] = TST_computeClassificationErrDNN(mTrainFeatures, mTrainTargets, NM_strNetParams, 0, 0, 0, 1, 1, 'ABSOLUTE_ERR_CALC', nBitfieldLength, vChunkLength, vOffset, CONFIG_strParams.eFeaturesMode);
%[nErr, vObtainedTrainTargets, vDesiredTrainTargets] = TST_computeClassificationErrDNN(mTrainBatchData, mTrainBatchTargets, NM_strNetParams, 0, 0, 0, 1, 1, 'ABSOLUTE_ERR_CALC', nBitfieldLength, vChunkLength, vOffset, CONFIG_strParams.eFeaturesMode);
%[nErr, vObtainedTrainTargets, vDesiredTrainTargets] = TST_computeClassificationErrDNN(mTestBatchData, mTestBatchTargets, NM_strNetParams, 0, 0, 0, 1, 1, 'ABSOLUTE_ERR_CALC', nBitfieldLength, vChunkLength, vOffset, CONFIG_strParams.eFeaturesMode);
%[nErr, vObtainedTargets, vDesiredTargets] = TST_computeClassificationErrDNN(mTestFeatures, mTestTargets, NM_strNetParams, 0, 0, 0, 1, 1, 'ABSOLUTE_ERR_CALC', nBitfieldLength, vChunkLength, vOffset, CONFIG_strParams.eFeaturesMode);
[nErr, vObtainedTargets, vDesiredTargets] = TST_computeClassificationErrDNN(mTestBatchData, mTestBatchTargets, NM_strNetParams, 0, 0, 0, 1, 1, 'ABSOLUTE_ERR_CALC', nBitfieldLength, vChunkLength, vOffset, CONFIG_strParams.eFeaturesMode);
%vTargetsDNN = vTargetsDNN';
% [I J]=max(mTrainTargets, [], 2);
% vTargets = J';
% vTrainTargets = vTargets;
[mConfusionMatrix, mNormalConfusionMatrix, vNumTrainExamplesPerClass, vAccuracyPerClass, nOverallAccuracy] = LM_buildConfusionMatrix(vDesiredTargets, vObtainedTargets);