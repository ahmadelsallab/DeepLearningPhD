% Function:
% Converts the input txt file into features and targets MATLAB vectors.
% Inputs:
% sTrainDataFile: The name of the .dat train file
% sTrainLabelsFile: The name of the .label train file
% sTestDataFile: The name of the .dat test file
% sTestLabelsFile: The name of the .label test file
% eFeaturesMode: Normal, Sparse
% nNumSupervisedExamplesPerClass: Number of supervised labels per class
% Output:
% mTrainFeatures, mTestFeatures: Matrix (nxm), where n is the number of examples and m is the features vector length
% mTrainTargets, mTestTargets: Matrix (nxl), where n is the number of examples and l is the number of target classes
function DCONV_convert20News(CONFIG_strParams)
        
    switch(CONFIG_strParams.eFeaturesMode)
        case 'Normal'

            load(CONFIG_strParams.sTrainDataFile);
            mTrainFeatures_ = full(spconvert(train));
            
            load(CONFIG_strParams.sTrainLabelsFile);
            
            %mTrainTargets = train;
            %mTrainTargets = zeros(size(train, 1), 20);
            %             mTrainTargets = zeros(size(train, 1), 20);
%             for i = 1 : size(train, 1)
%                 mTrainTargets(i, train(i)) = 1;
%             end

            mTrainTargets = [];
            mTrainFeatures = [];
            for i = 1 : size(train, 1)
                if(~isempty(find(train(i)==CONFIG_strParams.vSubClassTargets, 1)))
                    temp = zeros(1, length(CONFIG_strParams.vSubClassTargets));
                    temp(:, find(train(i)==CONFIG_strParams.vSubClassTargets, 1)) = 1;
                    mTrainTargets = [mTrainTargets; temp];
                    mTrainFeatures = [mTrainFeatures; mTrainFeatures_(i, :)];
                end
            end
            
            
            load(CONFIG_strParams.sTestDataFile);
            mTestFeatures_ = full(spconvert(test));
            load(CONFIG_strParams.sTestLabelsFile);
            %mTestTargets = test;
%             mTestTargets = zeros(size(test, 1), 20);
%             for i = 1 : size(test, 1)
%                 mTestTargets(i, test(i)) = 1;
%             end
            mTestTargets = [];
            mTestFeatures = [];
            for i = 1 : size(test, 1)
                if(~isempty(find(test(i)==CONFIG_strParams.vSubClassTargets, 1)))
                    temp = zeros(1, length(CONFIG_strParams.vSubClassTargets));
                    temp(:, find(test(i)==CONFIG_strParams.vSubClassTargets, 1)) = 1;
                    mTestTargets = [mTestTargets; temp];
                    mTestFeatures = [mTestFeatures; mTestFeatures_(i,:)];
                end
            end            



        case 'Sparse'
            load(CONFIG_strParams.sTrainDataFile);
            mTrainFeatures_ = spconvert(train);
            load(CONFIG_strParams.sTrainLabelsFile);
            %mTrainTargets = train;
%             mTrainTargets = zeros(size(train, 1), 20);
%             for i = 1 : size(train, 1)
%                 mTrainTargets(i, train(i)) = 1;
%             end
            mTrainTargets = [];
            mTrainFeatures = [];
            for i = 1 : size(train, 1)
                if(~isempty(find(train(i)==CONFIG_strParams.vSubClassTargets, 1)))
                    temp = zeros(1, length(CONFIG_strParams.vSubClassTargets));
                    temp(:, find(train(i)==CONFIG_strParams.vSubClassTargets, 1)) = 1;
                    mTrainTargets = [mTrainTargets; temp];
                    mTrainFeatures = [mTrainFeatures; mTrainFeatures_(i, :)];
                end
            end
            
            load(CONFIG_strParams.sTestDataFile);
            mTestFeatures_ = spconvert(test);
            load(CONFIG_strParams.sTestLabelsFile);
            %mTestTargets = test;

%             mTestTargets = zeros(size(test, 1), 20);
%             for i = 1 : size(test, 1)
%                 mTestTargets(i, test(i)) = 1;
%             end
            mTestTargets = [];
            mTestFeatures = [];
            for i = 1 : size(test, 1)
                if(~isempty(find(test(i)==CONFIG_strParams.vSubClassTargets, 1)))
                    temp = zeros(1, length(CONFIG_strParams.vSubClassTargets));
                    temp(:, find(test(i)==CONFIG_strParams.vSubClassTargets, 1)) = 1;
                    mTestTargets = [mTestTargets; temp];
                    mTestFeatures = [mTestFeatures; mTestFeatures_(i, :)];
                end
            end 
    end
       
    % Save to the input_data.mat workspace
    save(CONFIG_strParams.sInputDataWorkspace, '-v7.3', 'mTestFeatures', 'mTestTargets', 'mTrainFeatures', 'mTrainTargets');
end % end function