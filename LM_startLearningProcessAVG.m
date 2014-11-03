% Function:
% It starts the learning process of Average Classifier.
% Means are obtained for each class
% Classification is obtained by getting the nearest class label by
% computing the euclidean distance
% Inputs:
% CONFIG_strParams: The configuration parameters
% mTrainFeatures, mTrainTargets, mTestFeatures, mTestTargets: Used to train and test the
% Average Classifier on input data
% Output:
% None
function LM_startLearningProcessAVG(CONFIG_strParams,...
                                    mTrainFeatures, mTrainTargets, mTestFeatures, mTestTargets)

    LM_strLearningPrcocessPrvt.hFidLog = fopen(CONFIG_strParams.sLearnLogFile,'w');
    
    % Initialize TST structure
    TST_strPerformanceInfo = [];


    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Start training AVG...\n');
    fprintf(1,'Start training AVG...\n');
    
    [mAverageExamplePerClass] = CLS_trainAVG(mTrainFeatures, mTrainTargets);
    
    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Finished training AVG\n');
    fprintf(1,'Finished training AVG\n');        
	    
    % Compute train and test errors
    [TST_strPerformanceInfo.nTrainErr, TST_strPerformanceInfo.nTestErr, vTrainTargetsOut, vTestTargetsOut] =...
        TST_computeClassificationErrMAXENT(mTrainFeatures, mTrainTargets, mTestFeatures, mTestTargets, mAverageExamplePerClass);

    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'MaxEnt Test Error %d (out of %d)\n', TST_strPerformanceInfo.nMaxEntTestErr, size(mTestFeatures, 1));
    fprintf(1,'HMM Test Error %d (out of %d)\n', TST_strPerformanceInfo.nMaxEntTestErr, size(mTrainFeatures, 1));

       
    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'MaxEnt Test Accuracy %d \n', TST_strPerformanceInfo.nMaxEntTestAccuracy);
    fprintf(1,'MaxEnt Test Accuracy %d \n', TST_strPerformanceInfo.nMaxEntTestAccuracy);

    
    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'HMM Train Error %d (out of %d)\n', TST_strPerformanceInfo.nMaxEntTrainErr, size(mTrainFeatures, 1));
    fprintf(1,'HMM Train Error %d (out of %d)\n', TST_strPerformanceInfo.nHMMTrainErr, size(mTrainFeatures, 1));

    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'MaxEnt Train Accuracy %d \n', TST_strPerformanceInfo.nMaxEntTrainAccuracy);
    fprintf(1,'MaxEnt Train Accuracy %d \n', TST_strPerformanceInfo.nMaxEntTrainAccuracy);
    
    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Finished testing MaxEnt\n');
    fprintf(1,'Finished testing MaxEnt\n');

    % Save the params
    save(CONFIG_strParams.sNaiveBayesWorkSpaceFileName, 'mAverageExamplePerClass');
    
    % Build confusion matrix
    if(CONFIG_strParams.bBuildConfusionMatrix == 1)
        
        %%%%%%%%%%%%%% TRAIN CONFUSION MATRIX %%%%%%%%%%%%%%%%%%%%%%%%
        fprintf(1,'Start Train Building Confusion Matrix...\n');
        fprintf(LM_strLearningPrcocessPrvt.hFidLog, 'Start Train Building Confusion Matrix...\n');
        % Get the train target vector
        [I vTrainTargets]=max(mTrainTargets, [], 2);

        % Build confusion matrix
        [TST_strPerformanceInfo.mTrainConfusionMatrix, TST_strPerformanceInfo.mTrainNormalConfusionMatrix, TST_strPerformanceInfo.vTrainNumTrainExamplesPerClass,...
        TST_strPerformanceInfo.vTrainAccuracyPerClass, TST_strPerformanceInfo.nTrainOverallAccuracy] = LM_buildConfusionMatrix(vTrainTargets', vTrainTargetsOut);

        fprintf(1,'End Train Building Confusion Matrix\n');
        fprintf(LM_strLearningPrcocessPrvt.hFidLog, 'End Train Building Confusion Matrix\n');

        %%%%%%%%%%%%%% TEST CONFUSION MATRIX %%%%%%%%%%%%%%%%%%%%%%%%
        fprintf(1,'Start Test Building Confusion Matrix...\n');
        fprintf(LM_strLearningPrcocessPrvt.hFidLog, 'Start Test Building Confusion Matrix...\n');
        % Get the train target vector
        [I vTestTargets]=max(mTestTargets, [], 2);

        % Build confusion matrix
        [TST_strPerformanceInfo.mTestConfusionMatrix, TST_strPerformanceInfo.mTestNormalConfusionMatrix, TST_strPerformanceInfo.vTestNumTrainExamplesPerClass,...
        TST_strPerformanceInfo.vTestAccuracyPerClass, TST_strPerformanceInfo.nTestOverallAccuracy] = LM_buildConfusionMatrix(vTestTargets', vTestTargetsOut);

        fprintf(1,'End Test Building Confusion Matrix\n');
        fprintf(LM_strLearningPrcocessPrvt.hFidLog, 'End Test Building Confusion Matrix\n');
        
    end % end if
    
    % Save the current configuration in the error performance workspace
    save(CONFIG_strParams.sNameofErrWorkspace, 'TST_strPerformanceInfo');
    
	% Close the log file
    fclose(LM_strLearningPrcocessPrvt.hFidLog);
	
end % end function