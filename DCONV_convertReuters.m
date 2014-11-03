% Function:
% Converts the input txt file into features and targets MATLAB vectors.
% Inputs:
% sTrainFile: The name of the .mat train file
% sTestFile: The name of the .mat test file
% eFeaturesMode: Normal, Sparse
% nNumSupervisedExamplesPerClass: Number of supervised labels per class
% Output:
% mTrainFeatures, mTestFeatures: Matrix (nxm), where n is the number of examples and m is the features vector length
% mTrainTargets, mTestTargets: Matrix (nxl), where n is the number of examples and l is the number of target classes
function DCONV_convertReuters(CONFIG_strParams)
    
    load(CONFIG_strParams.sTrainFile);
    load(CONFIG_strParams.sTestFile);
    %nTrainExamples = 1000;
    %nTestExamples = 100;
    nNumTargets = max(labels);
    
    switch(CONFIG_strParams.eFeaturesMode)
        case 'Normal'
            % Convert sparse into normal matrices

            %mTrainFeatures = (full(counts(:,1:nTrainExamples)))';
            %mTestFeatures = (full(counts_test(:,1:nTestExamples)))';
            %mTrainTargets_ = labels(1:nTrainExamples);
            %mTestTargets_ = labels_test(1:nTestExamples);
            mTrainFeatures = (full(counts))';
            mTestFeatures = (full(counts_test))';
            %mTrainTargets = labels;
			mTrainTargets = zeros(size(labels, 1), 23);
            for i = 1 : size(labels, 1)
                mTrainTargets(i, labels(i)) = 1;
            end
            %mTestTargets = labels_test;		
			mTestTargets = zeros(size(labels_test, 1), 23);
            for i = 1 : size(labels_test, 1)
                mTestTargets(i, labels_test(i)) = 1;
            end

        case 'Sparse'
            mTrainFeatures = counts';
            mTestFeatures = counts_test';
            %mTrainTargets = labels;
			mTrainTargets = zeros(size(labels, 1), 23);
            for i = 1 : size(labels, 1)
                mTrainTargets(i, labels(i)) = 1;
            end
            %mTestTargets = labels_test;		
			mTestTargets = zeros(size(labels_test, 1), 23);
            for i = 1 : size(labels_test, 1)
                mTestTargets(i, labels_test(i)) = 1;
            end

    end

    % Save to the input_data.mat workspace
    save(CONFIG_strParams.sInputDataWorkspace, '-v7.3', 'mTestFeatures', 'mTestTargets', 'mTrainFeatures', 'mTrainTargets');
end % end function