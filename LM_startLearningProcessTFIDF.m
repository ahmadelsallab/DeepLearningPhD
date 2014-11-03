% Function:
% It starts the learning process of TF-IDF Classifier.
% Inputs:
% CONFIG_strParams: The configuration parameters
% mTrainFeatures, mTrainTargets, mTestFeatures, mTestTargets: Used to train and test the
% classifier on input data
% Output:
% None
function LM_startLearningProcessTFIDF(CONFIG_strParams,...
                                      mTrainFeatures, mTrainTargets, mTestFeatures, mTestTargets)

    LM_strLearningPrcocessPrvt.hFidLog = fopen(CONFIG_strParams.sLearnLogFile,'w');
    
    % Initialize TST structure
    TST_strPerformanceInfo = [];


    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Start training TF-IDF...\n');
    fprintf(1,'Start training TF-IDF...\n');   

    TFIDF_clsParams = CLS_trainTFIDF(mTrainFeatures, mTrainTargets);
    
    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Finished training TF-IDF\n');
    fprintf(1,'Finished training TF-IDF\n');        
	    
    % Compute train and test errors
    [TST_strPerformanceInfo.nTrainErr, TST_strPerformanceInfo.nTestErr, TST_strPerformanceInfo.vTrainTargetsOut, TST_strPerformanceInfo.vTestTargetsOut] =...
        TST_computeClassificationErrTFIDF(mTrainFeatures, mTrainTargets, mTestFeatures, mTestTargets, TFIDF_clsParams);
        
    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'TF-IDF Test Error %d (out of %d)\n', TST_strPerformanceInfo.nTestErr, size(mTestFeatures, 1));
    fprintf(1,'TF-IDF Test Error %d (out of %d)\n', TST_strPerformanceInfo.nTestErr, size(mTestFeatures, 1));
    
    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'HMM Train Error %d (out of %d)\n', TST_strPerformanceInfo.nTrainErr, size(mTrainFeatures, 1));
    fprintf(1,'TF-IDF Train Error %d (out of %d)\n', TST_strPerformanceInfo.nTrainErr, size(mTrainFeatures, 1));
    
    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Finished testing TF-IDF\n');
    fprintf(1,'Finished testing TF-IDF\n');

    % Save the params
    save(CONFIG_strParams.sTFIDFWorkSpaceFileName, 'TFIDF_clsParams');
    
    % Build confusion matrix
    if(CONFIG_strParams.bBuildConfusionMatrix == 1)
        
        %%%%%%%%%%%%%% TRAIN CONFUSION MATRIX %%%%%%%%%%%%%%%%%%%%%%%%
        fprintf(1,'Start Train Building Confusion Matrix...\n');
        fprintf(LM_strLearningPrcocessPrvt.hFidLog, 'Start Train Building Confusion Matrix...\n');
        % Get the train target vector
        [I vTrainTargets]=max(mTrainTargets, [], 2);

        % Build confusion matrix
        [TST_strPerformanceInfo.mTrainConfusionMatrix, TST_strPerformanceInfo.mTrainNormalConfusionMatrix, TST_strPerformanceInfo.vTrainNumTrainExamplesPerClass,...
        TST_strPerformanceInfo.vTrainAccuracyPerClass, TST_strPerformanceInfo.nTrainOverallAccuracy] = LM_buildConfusionMatrix(vTrainTargets', TST_strPerformanceInfo.vTrainTargetsOut);

        fprintf(1,'End Train Building Confusion Matrix\n');
        fprintf(LM_strLearningPrcocessPrvt.hFidLog, 'End Train Building Confusion Matrix\n');

        %%%%%%%%%%%%%% TEST CONFUSION MATRIX %%%%%%%%%%%%%%%%%%%%%%%%
        fprintf(1,'Start Test Building Confusion Matrix...\n');
        fprintf(LM_strLearningPrcocessPrvt.hFidLog, 'Start Test Building Confusion Matrix...\n');
        % Get the train target vector
        [I vTestTargets]=max(mTestTargets, [], 2);

        % Build confusion matrix
        [TST_strPerformanceInfo.mTestConfusionMatrix, TST_strPerformanceInfo.mTestNormalConfusionMatrix, TST_strPerformanceInfo.vTestNumTrainExamplesPerClass,...
        TST_strPerformanceInfo.vTestAccuracyPerClass, TST_strPerformanceInfo.nTestOverallAccuracy] = LM_buildConfusionMatrix(vTestTargets', TST_strPerformanceInfo.vTestTargetsOut);

        fprintf(1,'End Test Building Confusion Matrix\n');
        fprintf(LM_strLearningPrcocessPrvt.hFidLog, 'End Test Building Confusion Matrix\n');
        
    end % end if
    
    % Save the current configuration in the error performance workspace
    save(CONFIG_strParams.sNameofErrWorkspace, 'TST_strPerformanceInfo');
    
	% Close the log file
    fclose(LM_strLearningPrcocessPrvt.hFidLog);
	
end % end function