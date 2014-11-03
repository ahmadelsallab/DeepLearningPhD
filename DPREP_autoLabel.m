% Function:
% Converts the test and train sets into batches
% Inputs:
% mTrainTargets: Train targets. Matrix (nxl), where n is the number of examples and l is the number of target classes
% bAutoLabel: Flag to indicate if automatic (unsupervised) labeling is used
% mAutoLabels: The automatic labels
% Output:
% mTrainTargets: Train targets. Matrix (nxl), where n is the number of examples and l is the number of target classes with Auto labels
function [mTrainTargets, mAutoLabels] = DPREP_autoLabel(mTrainTargets, CONFIG_strParams)

    nNumTargets = size(mTrainTargets, 2);
	
	% Load auto label data if enabled
	if(CONFIG_strParams.bAutoLabel == 1)
		load(CONFIG_strParams.sAutoLabelWorkSpaceFileName, CONFIG_strParams.sAutoLabelNewLabelsVarName);
		mAutoLabels = eval(CONFIG_strParams.sAutoLabelNewLabelsVarName);
				
		%mTrainTargets = mTrainTargets(1 : size(mTrainTargets, 1) * (CONFIG_strParams.nPrecentManuallyLabeledExamples / 100), :);
		
		for i = size(mTrainTargets, 1) * ((CONFIG_strParams.nPrecentManuallyLabeledExamples / 100) + 1) : size(mAutoLabels, 2)
			for j = 1 : nNumTargets
				if(mAutoLabels(i)==j)
					mTrainTargets(i,j) = 1;
				else
					mTrainTargets(i,j) = 0;
				end
			end
		end
	else
		mAutoLabels = [];
	end

end % end function