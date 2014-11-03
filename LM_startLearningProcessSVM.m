% Function:
% Train and test SVM classifier.
% Inputs:
% CONFIG_strParams: Configuration parameters
% mTestFeatures: Test features. Matrix (nxm), where n is the number of examples and m is the features vector length
% mTestTargets: Test targetss. Matrix (nxl), where n is the number of examples and l is the number of target classes
% mTrainFeatures: Train features. Matrix (nxm), where n is the number of examples and m is the features vector length
% mTrainTargets: Train targets. Matrix (nxl), where n is the number of examples and l is the number of target classes
% Output:
% None. The TST_strErrorPerformance structure is saved in the corresponding .mat file and the error perf. figure is displayed.
function LM_startLearningProcessSVM(CONFIG_strParams, mTestFeatures, mTestTargets, mTrainFeatures, mTrainTargets)

	% Open log file
	LM_strLearningPrcocessPrvt.hFidLog = fopen(CONFIG_strParams.sLearnLogFile,'w');

	LM_strLearningPrcocessPrvt.nNumTargets = size(mTrainTargets, 2); % LM_strLearningPrcocessPrvt.nNumTargets = num_svm_classifiers
	LM_strLearningPrcocessPrvt.nNumTrainExamples = size(mTrainFeatures, 1);
	LM_strLearningPrcocessPrvt.nNumTestExamples = size(mTestFeatures, 1);

	fprintf(1, 'Training SVMs...\n');
	%fprintf(LM_strLearningPrcocessPrvt.hFidLog, 'Training SVMs...\n');

	% Train LM_strLearningPrcocessPrvt.nNumTargets SVM's
	for j = 1 : LM_strLearningPrcocessPrvt.nNumTargets
		for i = 1 : LM_strLearningPrcocessPrvt.nNumTrainExamples
			fprintf(1, 'Convert example %d to train SVM %d \n', i, j);
			target = find(mTrainTargets(i,:) == 1);
			if(target == j)
				labels(i) = 1;
			else
				labels(i) = 0;
			end
		end
		labels = labels';
		fprintf(1, 'Start training SVM %d \n', j);
		SVM_strParams(j) = svmtrain(mTrainFeatures, labels, 'kernel_function', CONFIG_strParams.eKernelFunction);
		fprintf(1, 'Finished training SVM %d \n', j);
	end

	fprintf(1, 'Training SVMs done successfuly\n');
	%fprintf(LM_strLearningPrcocessPrvt.hFidLog, 'Training SVMs done successfuly\n');

	save(CONFIG_strParams.sSVMWorkSpaceFileName, 'SVM_strParams');
	
	% Train error
	[TST_strPerformanceInfo.nTrainErr, TST_strPerformanceInfo.nTrainConfusionErr, TST_strPerformanceInfo.nTrainErrRate, TST_strPerformanceInfo.nTrainConfusionErrRate] =...
		TST_computeClassificationErrSVM(LM_strLearningPrcocessPrvt.hFidLog, mTrainFeatures, mTrainTargets, SVM_strParams);

	 
	% Test SVM
	[TST_strPerformanceInfo.nTestErr, TST_strPerformanceInfo.nTestConfusionErr, TST_strPerformanceInfo.nTestErrRate, TST_strPerformanceInfo.nTestConfusionErrRate] =...
		TST_computeClassificationErrSVM(LM_strLearningPrcocessPrvt.hFidLog, mTestFeatures, mTestTargets, SVM_strParams);

    
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
 
 end