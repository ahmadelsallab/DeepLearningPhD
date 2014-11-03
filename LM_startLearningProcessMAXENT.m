% Function:
% It starts the learning process of MaxEnt (maximum entropy classifier)
% Inputs:
% CONFIG_strParams: The configuration parameters
% mTrainFeatures, mTrainTargets, mTestFeatures, mTestTargets: Used to train and test the
% MaxEnt on input data
% Output:
% None
function LM_startLearningProcessMAXENT(CONFIG_strParams,...
                                        mTrainFeatures, mTrainTargets, mTestFeatures, mTestTargets)

    LM_strLearningPrcocessPrvt.hFidLog = fopen(CONFIG_strParams.sLearnLogFile,'w');
    
    % Initialize TST structure
    TST_strPerformanceInfo = [];


    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Start training MaxEnt...\n');
    fprintf(1,'Start training MaxEnt...\n');
    
    [MAXENT_clsParams] = CLS_trainMAXENT(mTrainFeatures, mTrainTargets);
    
    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Finished training MaxEnt\n');
    fprintf(1,'Finished training MaxEnt\n');        
	    
    % Compute train and test errors
    [TST_strPerformanceInfo.nMaxEntTrainErr, TST_strPerformanceInfo.nMaxEntTestErr, TST_strPerformanceInfo.nMaxEntTrainAccuracy, TST_strPerformanceInfo.nMaxEntTestAccuracy] =...
        TST_computeClassificationErrMAXENT(mTrainFeatures, mTrainTargets, mTestFeatures, mTestTargets, MAXENT_clsParams);

    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'MaxEnt Test Error %d (out of %d)\n', TST_strPerformanceInfo.nMaxEntTestErr, size(mTestFeatures, 1));
    fprintf(1,'MaxEnt Test Error %d (out of %d)\n', TST_strPerformanceInfo.nMaxEntTestErr, size(mTestFeatures, 1));

       
    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'MaxEnt Test Accuracy %d \n', TST_strPerformanceInfo.nMaxEntTestAccuracy);
    fprintf(1,'MaxEnt Test Accuracy %d \n', TST_strPerformanceInfo.nMaxEntTestAccuracy);

    
    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'MaxEnt Train Error %d (out of %d)\n', TST_strPerformanceInfo.nMaxEntTrainErr, size(mTrainFeatures, 1));
    fprintf(1,'MaxEnt Train Error %d (out of %d)\n', TST_strPerformanceInfo.nMaxEntTrainErr, size(mTrainFeatures, 1));

    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'MaxEnt Train Accuracy %d \n', TST_strPerformanceInfo.nMaxEntTrainAccuracy);
    fprintf(1,'MaxEnt Train Accuracy %d \n', TST_strPerformanceInfo.nMaxEntTrainAccuracy);
    
    fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Finished testing MaxEnt\n');
    fprintf(1,'Finished testing MaxEnt\n');

    % Save the params
    save(CONFIG_strParams.sMaxEntWorkSpaceFileName, 'MAXENT_clsParams');
    
    % Build confusion matrix
    if(CONFIG_strParams.bBuildConfusionMatrix == 1)
        
        %%%%%%%%%%%%%% TRAIN CONFUSION MATRIX %%%%%%%%%%%%%%%%%%%%%%%%
        fprintf(1,'Start Train Building Confusion Matrix...\n');
        fprintf(LM_strLearningPrcocessPrvt.hFidLog, 'Start Train Building Confusion Matrix...\n');
        % Get the train target vector
        [I vTrainTargets]=max(mTrainTargets, [], 2);

        % Obtain the output train targets
        vTrainTargetsOut = map(MAXENT_clsParams, mTrainFeatures);

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

        % Obtain the output train targets
        vTestTargetsOut = map(MAXENT_clsParams, mTestFeatures);

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