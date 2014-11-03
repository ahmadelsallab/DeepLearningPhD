% Function:
% Makes feed forward in the net and reports the error.
% Inputs:
% mData: Input data. Matrix (nxm), where n is the number of examples and m is the features vector length
% mTargets: Associated targets. Vector (nxl), where n is the number of examples and l is the number of target classes
% NM_strNetParams: Net parameters to feed forward
% SVM_strParams: SVM parameters
% HMM_strParams: HMM transition and emission probs
% CONFIG_strParams: Configurations parameters
% Output:
% nErr: number of misclassified examples
function [nErr, nConfusionErr, nErrRate, nConfusionErrRate] = TST_computeClassificationErr(hFidLog, mData, mTargets, NM_strNetParams, SVM_strParams, HMM_strParams, CONFIG_strParams)

	switch(CONFIG_strParams.eClassifierType)
		case 'DNN'
			[nErr, vTargetOut] = TST_computeClassificationErrDNN(mData, mTargets, NM_strNetParams, CONFIG_strParams.bMapping,...
																 CONFIG_strParams.eMappingDirection, CONFIG_strParams.eMappingMode,...
																 1, 1, 'EPOCH_ERR_CALC');
			nConfusionErr = 0;
			nErrRate = nErr / size(mData ,1);
			nConfusionErrRate = 0;			
		case 'SVM'						
			[nErr, nConfusionErr, nErrRate, nConfusionErrRate] =...
				TST_computeClassificationErrSVM(hFidLog, mData, mTargets, SVM_strParams);
        case 'DNN_HMM'						
			[nErr] = TST_computeClassificationErrDNN_HMM(mTargets, HMM_strParams);% Obsolete
	end % end switch
	
	fprintf(1, 'Test error rate %d \n', nErrRate);
	fprintf(1, 'Test confusion error rate %d \n', nConfusionErrRate);
	fprintf(hFidLog, 'Test error rate %d \n', nErrRate);
	fprintf(hFidLog, 'Test confusion error rate %d \n', nConfusionErrRate);

end % end function