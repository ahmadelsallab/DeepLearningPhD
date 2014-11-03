% Function:
% Makes feed forward in the net and reports the error.
% Inputs:
% mData: Input data. Matrix (nxm), where n is the number of examples and m is the features vector length
% mTargets: Associated targets. Vector (nxl), where n is the number of examples and l is the number of target classes
% NM_strNetParams: Net parameters to feed forward
% SVM_strParams: SVM parameters
% CONFIG_strParams: Configurations parameters
% hFidLog: Handle to the log file
% Output:
% nErr: number of misclassified examples
% nConfusionErr: multiple decisions, one of them might be right, so the decision is confused
% nErrRate: Percentage of error to total examples
% nConfusionErrRate: Percentage of confusion errors to total examples
function [nErr, nConfusionErr, nErrRate, nConfusionErrRate] = TST_computeClassificationErrSVM(hFidLog, mData, mTargets, SVM_strParams)

	% Initialize counters
	nErr = 0;
	nConfusionErr = 0;
	
	% Get the number of examples to loop on
	nNumExamples = size(mData, 1);
	
	% nNumTargets = num_svm_classifiers = number of SVM's to be used
	nNumTargets = size(mTargets, 2);
	
	for i = 1 : nNumExamples
		
		fprintf(1, 'Test example %d \n', i);
		% Get decisions of each SVM
		for j = 1 : nNumTargets
			decision(j) = svmclassify(SVM_strParams(j), mTestFeatures(i,:));
		end
		
		% Get class = decision of SVM == 1
		class = find(decision == 1);
		
		
		% Misclassified example if:
		% 1- More than 1 SVM classifies it to belong to its class
		% 2- test_target != SVM decision
		if(size(class,2) ~= 1)
			nConfusionErr = nConfusionErr + 1;
			nErr = nErr + 1;
			fprintf(1, 'Test example %d confused between %d \n', i, class);
			fprintf(hFidLog, 'Test example %d confused between %d \n', i, class);
			fprintf(1, 'Test err counter %d \n', nErr);
			fprintf(hFidLog, 'Test err counter %d \n', nErr);
		else if(class ~= find(mTestTargets(i,:)==1))
				nErr = nErr + 1;
				fprintf(1, 'Test example %d misclassified \n', i);
				fprintf(hFidLog, 'Test example %d misclassified \n', i);
				fprintf(1, 'Test err counter %d \n', nErr);
				fprintf(hFidLog, 'Test err counter %d \n', nErr);
			else
				fprintf(1, 'Test example %d correctly classified \n', i);
				fprintf(hFidLog, 'Test example %d correctly classified \n', i);
			end
		end
		
	end

	nErrRate = nErr/nNumExamples * 100;
	nConfusionErrRate = nConfusionErr/nNumExamples * 100;

end % end function