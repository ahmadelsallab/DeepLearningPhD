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
function CLS_trainAndTestSVM(CONFIG_strParams, TST_strPerformanceInfo, hFidLog, mTestFeatures, mTestTargets, mTrainFeatures, mTrainTargets)



	nNumTargets = size(mTrainTargets, 2); % nNumTargets = num_svm_classifiers
	nNumTrainExamples = size(mTrainFeatures, 1);
	nNumTestExamples = size(mTestFeatures, 1);

	fprintf(1, 'Training SVMs...\n');
	%fprintf(hFidLog, 'Training SVMs...\n');

	% Train nNumTargets SVM's
	for j = 1 : nNumTargets
		for i = 1 : nNumTrainExamples
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
	%fprintf(hFidLog, 'Training SVMs done successfuly\n');

	save(CONFIG_strParams.sSVMWorkSpaceFileName, 'SVM_strParams');
	
	% Train error
	[TST_strPerformanceInfo.nSVMTrainErr, TST_strPerformanceInfo.nTrainConfusionErr, TST_strPerformanceInfo.nTrainErrRate, TST_strPerformanceInfo.nTrainConfusionErrRate] =...
		TST_computeClassificationErrSVM(hFidLog, mTrainFeatures, mTrainTargets, SVM_strParams);

	 
	% Test SVM
	[TST_strPerformanceInfo.nSVMTestErr, TST_strPerformanceInfo.nTestConfusionErr, TST_strPerformanceInfo.nTestErrRate, TST_strPerformanceInfo.nTestConfusionErrRate] =...
		TST_computeClassificationErrSVM(hFidLog, mTestFeatures, mTestTargets, SVM_strParams);

	% Save the current configuration in the error performance workspace
	save(CONFIG_strParams.sNameofErrWorkspace, 'TST_strPerformanceInfo');
 
 end