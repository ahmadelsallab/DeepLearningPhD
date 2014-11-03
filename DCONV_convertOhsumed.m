% Function:
% Converts the input txt file into features and targets MATLAB vectors.
% Inputs:
% sDataFile: The name of the .dat train file
% sLabelsFile: The name of the .label train file
% sCriteria: The splitting criteria
% nTrainToTestFactor: The ratio of splitting train to test setss
% eFeaturesMode: Normal, Sparse
% nNumSupervisedExamplesPerClass: Number of supervised labels per class
% Output:
% mTrainFeatures, mTestFeatures: Matrix (nxm), where n is the number of examples and m is the features vector length
% mTrainTargets, mTestTargets: Matrix (nxl), where n is the number of examples and l is the number of target classes
function DCONV_convertOhsumed(CONFIG_strParams)
    
    % Load counts and labels
    load(CONFIG_strParams.sDataLabelsFile);  
    
    switch(CONFIG_strParams.eFeaturesMode)
        case 'Normal'
            % Convert sparse into normal matrices
            mFeaturesSparse = counts';
            mFeatures = full(mFeaturesSparse);            
            %mTargets = labels;
            mTargets = zeros(size(labels, 1), 23);
            for i = 1 : size(labels, 1)
                mTrainTargets(i, labels(i)) = 1;
            end


        case 'Sparse'
            mFeatures = counts';            
            %mTargets = labels;
            mTargets = zeros(size(labels, 1), 23);
            for i = 1 : size(labels, 1)
                mTargets(i, labels(i)) = 1;
            end


    end
    
    % Split into train and test
    [mTestFeatures, mTestTargets, mTrainFeatures, mTrainTargets] = TTS_formTrainTestSets(mFeatures, mTargets, CONFIG_strParams.sSplitCriteria, CONFIG_strParams.nTrainToTestFactor);
    
    % Save to the input_data.mat workspace
    save(CONFIG_strParams.sInputDataWorkspace, '-v7.3', 'mTestFeatures', 'mTestTargets', 'mTrainFeatures', 'mTrainTargets');

end % end function