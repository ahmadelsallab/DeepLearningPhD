% Function:
% The main entry point to train and test Generic Deep ANN
% Inputs:
% sDefaultPath: String with the default path of the input txt file
% sFeaturesFileName: String of the input txt file
% Output:
% None
function MAIN_trainAndClassify(CONFIG_strParams)    
    
    fprintf(1, 'Converting input files...\n');
	% Check the dataset used
	switch(CONFIG_strParams.sDataset)
		case 'MNIST'
			% The output of conversion is saved in CONFIG_strParams.sInputDataWorkspace
			DCONV_convertMNIST(CONFIG_strParams);

		case '20News'
			% The output of conversion is saved in CONFIG_strParams.sInputDataWorkspace
			DCONV_convert20News(CONFIG_strParams);

		case 'Ohsumed'
			% The output of conversion is saved in CONFIG_strParams.sInputDataWorkspace
			DCONV_convertOhsumed(CONFIG_strParams);
		
		case 'Reuters'
			% The output of conversion is saved in CONFIG_strParams.sInputDataWorkspace
			DCONV_convertReuters(CONFIG_strParams);			
		
		case 'Diacritization'
			% Do nothing, a C# preprocessor has run and provided the required CONFIG_strParams.sInputDataWorkspace = input_data.mat
		otherwise
			% Do nothing
	end
	
	if(CONFIG_strParams.bReduceTrainingSetSize)
		fprintf(1, 'Reducing training set to %d percent...\n', CONFIG_strParams.nDesiredTrainSetSizePercent);
        load(CONFIG_strParams.sInputDataWorkspace);

		mTrainFeatures = mTrainFeatures(1 : size(mTrainFeatures, 1) * (CONFIG_strParams.nDesiredTrainSetSizePercent / 100), :);
		mTrainTargets = mTrainTargets(1 : size(mTrainTargets, 1) * (CONFIG_strParams.nDesiredTrainSetSizePercent / 100), :);
		
		save(CONFIG_strParams.sInputDataWorkspace, '-v7.3', 'mTestFeatures', 'mTestTargets', 'mTrainFeatures', 'mTrainTargets');
		
		fprintf(1, 'Reduction done successfuly\n');
	end	
    
    switch (CONFIG_strParams.sInputFormat)
        case 'MATLAB'
            % Convert raw data to matlab vars
            [DPREP_strData.mFeatures, DPREP_strData.mTargets] = DCONV_convertMatlabInput();
			
			% Save converted data
			save(CONFIG_strParams.sInputDataWorkspace, '-struct', 'DPREP_strData', 'mFeatures', 'mTargets');
        case 'TxtFile'
            % Convert raw data to matlab vars
            [DPREP_strData.mFeatures, DPREP_strData.mTargets, DPREP_strData.nBitfieldLength, DPREP_strData.vChunkLength, DPREP_strData.vOffset] = DCONV_convert(CONFIG_strParams.fullRawDataFileName, CONFIG_strParams.eFeaturesMode);
			% Save converted data
			save(CONFIG_strParams.sInputDataWorkspace, '-struct', 'DPREP_strData', 'mFeatures', 'mTargets');

        case 'MatlabWorkspace'
            load(CONFIG_strParams.sInputDataWorkspace);
            DPREP_strData.mTargets = mTargets;
            DPREP_strData.mFeatures = mFeatures;
            if(strcmp(CONFIG_strParams.eFeaturesMode, 'Raw'))
                DPREP_strData.nBitfieldLength = nBitfieldLength;
                DPREP_strData.vChunkLength = vChunkLength;
                DPREP_strData.vOffset = vOffset;

                clear mTargets mFeatures nBitfieldLength vChunkLength vOffset;

            else
                DPREP_strData.nBitfieldLength = 0;
                DPREP_strData.vChunkLength = [];
                DPREP_strData.vOffset = [];

                clear mTargets mFeatures;

            end
            
		case 'MatlabWorkspaceReadyTestTrainSplit'
            load(CONFIG_strParams.sInputDataWorkspace);
			nNumPhases = floor(CONFIG_strParams.nFinalFirstLayerWidth/CONFIG_strParams.vInitialLayersWidths(1));
			if(CONFIG_strParams.bDoubleTrainingSetSizeWithMapping == 1)							
				nReducedSize = floor(size(mTrainTargets, 1) / nNumPhases);
                fprintf(1, 'Reducing traininig size to %d...\n', nReducedSize);
				mTrainFeatures = mTrainFeatures(1 : nReducedSize, :);
				mTrainTargets = mTrainTargets(1 : nReducedSize, :);
				save('input_data_reduced');
				fprintf(1, 'Reduction done successfuly\n');
			end
            if(strcmp(CONFIG_strParams.eFeaturesMode, 'Raw'))
                DPREP_strData.nBitfieldLength = nBitfieldLength;
                DPREP_strData.vChunkLength = vChunkLength;
                DPREP_strData.vOffset = vOffset;
                clear nBitfieldLength vChunkLength vOffset;
            
            else
                DPREP_strData.nBitfieldLength = 0;
                DPREP_strData.vChunkLength = [];
                DPREP_strData.vOffset = [];
                
            end
        otherwise
            % Convert raw data to matlab vars
            [DPREP_strData.mFeatures, DPREP_strData.mTargets, DPREP_strData.nBitfieldLength DPREP_strData.vChunkLength, DPREP_strData.vOffset] = DCONV_convert(CONFIG_strParams.fullRawDataFileName, CONFIG_strParams.eFeaturesMode);          
    end

    
    fprintf(1, 'Conversion done successfuly\n');
    
	fprintf(1, 'Splitting dataset into train and test sets...\n');
	
	switch (CONFIG_strParams.sInputFormat)
		case 'MatlabWorkspaceReadyTestTrainSplit'
			%load(CONFIG_strParams.sInputDataWorkspace);
			DPREP_strData.mTestFeatures = mTestFeatures;
			DPREP_strData.mTestTargets = mTestTargets;
			DPREP_strData.mTrainFeatures = mTrainFeatures;
			DPREP_strData.mTrainTargets = mTrainTargets;
            clear mTestFeatures mTestTargets mTrainFeatures mTrainTargets;
        otherwise
			% Split into train and test sets
			[DPREP_strData.mTestFeatures, DPREP_strData.mTestTargets, DPREP_strData.mTrainFeatures, DPREP_strData.mTrainTargets] =... 
				TTS_formTrainTestSets(DPREP_strData.mFeatures,...
									  DPREP_strData.mTargets,...
									  CONFIG_strParams.sSplitCriteria,...
									  CONFIG_strParams.nTrainToTestFactor);
                
                %Save split data
                save(CONFIG_strParams.sInputDataWorkspace, '-struct', 'DPREP_strData', '-append',...
                     'mTestFeatures',...
                     'mTestTargets',...
                     'mTrainFeatures',...
                     'mTrainTargets');
                
                if(strcmp(CONFIG_strParams.sMemorySavingMode, 'ON'))
                    % clear DPREP_strData.mFeatures DPREP_strData.mTargets;
                    DPREP_strData.mFeatures = [];
                    DPREP_strData.mTargets = [];
                end
         
    end
    
	fprintf(1, 'Splitting done successfuly\n');

    % Load auto label data if enabled
 	[DPREP_strData.mTrainTargets, DPREP_strData.mAutoLabels] = DPREP_autoLabel(DPREP_strData.mTrainTargets, CONFIG_strParams);
 
	fprintf(1, 'Start learning process\n');
	switch(CONFIG_strParams.eClassifierType)
		case 'DNN'
			fprintf(1, 'Making train and test batches...\n');
			% Make train and test batches
			[DPREP_strData.mTrainBatchTargets, DPREP_strData.mTrainBatchData, DPREP_strData.mTestBatchTargets, DPREP_strData.mTestBatchData] =... 
				BM_makeBatches(DPREP_strData.mTestFeatures,...
							   DPREP_strData.mTestTargets,...
							   DPREP_strData.mTrainFeatures,...
							   DPREP_strData.mTrainTargets,...
							   CONFIG_strParams.nMaxFeaturesRange,...
							   CONFIG_strParams.bAutoLabel,...
							   DPREP_strData.mAutoLabels,...
							   CONFIG_strParams.nBatchSize);
			fprintf(1, 'Batches made successfuly\n');

			% Save Batch data
			save(CONFIG_strParams.sInputDataWorkspace, '-struct', 'DPREP_strData', '-append',... 
				'mTrainBatchTargets',... 
				'mTrainBatchData',...
				'mTestBatchTargets',...
				'mTestBatchData');

			if(strcmp(CONFIG_strParams.sMemorySavingMode, 'ON'))
				
				DPREP_strData.mTestFeatures = [];
				DPREP_strData.mTestTargets = [];
				DPREP_strData.mTrainFeatures = [];
				DPREP_strData.mTrainTargets = [];
				DPREP_strData.mAutoLabels = [];
			end
			
% DPREP_strData.mTrainBatchTargets = mTrainBatchTargets;
% clear mTrainBatchTargets;
% DPREP_strData.mTrainBatchData = mTrainBatchData;
% clear mTrainBatchData;
% DPREP_strData.mTestBatchTargets = mTestBatchTargets;
% clear mTestBatchTargets;
% DPREP_strData.mTestBatchData = mTestBatchData;
% clear mTestBatchData;
			
			% Start the learning process
			LM_startLearningProcessDNN(CONFIG_strParams,...
                                       DPREP_strData.mTestFeatures, DPREP_strData.mTestTargets, DPREP_strData.mTrainFeatures, DPREP_strData.mTrainTargets,...
                                       DPREP_strData.mTrainBatchTargets, DPREP_strData.mTrainBatchData, DPREP_strData.mTestBatchTargets, DPREP_strData.mTestBatchData,...
                                       DPREP_strData.nBitfieldLength, DPREP_strData.vChunkLength, DPREP_strData.vOffset, DPREP_strData.mAutoLabels);
			                  
		case 'SVM'
			% Start the learning process
			LM_startLearningProcessSVM(CONFIG_strParams,...
								       DPREP_strData.mTestFeatures,...
									   DPREP_strData.mTestTargets,...
									   DPREP_strData.mTrainFeatures,...
									   DPREP_strData.mTrainTargets);
        case 'DNN_HMM'
            
            fprintf(1, 'Train DNN...\n');
            
			fprintf(1, 'Making train and test batches...\n');
			% Make train and test batches
			[DPREP_strData.mTrainBatchTargets, DPREP_strData.mTrainBatchData, DPREP_strData.mTestBatchTargets, DPREP_strData.mTestBatchData] =... 
				BM_makeBatches(DPREP_strData.mTestFeatures,...
							   DPREP_strData.mTestTargets,...
							   DPREP_strData.mTrainFeatures,...
							   DPREP_strData.mTrainTargets,...
							   CONFIG_strParams.nMaxFeaturesRange,...
							   CONFIG_strParams.bAutoLabel,...
							   DPREP_strData.mAutoLabels,...
							   CONFIG_strParams.nBatchSize);
			fprintf(1, 'Batches made successfuly\n');
            
			% Save Batch data
			save(CONFIG_strParams.sInputDataWorkspace, '-struct', 'DPREP_strData', '-append',... 
				'mTrainBatchTargets',... 
				'mTrainBatchData',...
				'mTestBatchTargets',...
				'mTestBatchData');

			if(strcmp(CONFIG_strParams.sMemorySavingMode, 'ON'))
				
				DPREP_strData.mTestFeatures = [];
				DPREP_strData.mTestTargets = [];
				DPREP_strData.mTrainFeatures = [];
				DPREP_strData.mTrainTargets = [];
				DPREP_strData.mAutoLabels = [];
            end
            
%             %load 'input_data';
%             DPREP_strData.mTrainBatchTargets = mTrainBatchTargets;
%             clear mTrainBatchTargets;
%             DPREP_strData.mTrainBatchData = mTrainBatchData;
%             clear mTrainBatchData;
%             DPREP_strData.mTestBatchTargets = mTestBatchTargets;
%             clear mTestBatchTargets;
%             DPREP_strData.mTestBatchData = mTestBatchData;
%             clear mTestBatchData;
			
%             load 'input_data';
%             DPREP_strData.mTrainBatchTargets = mTrainBatchTargets;
%             DPREP_strData.mTrainBatchData = mTrainBatchData;
%             DPREP_strData.mTestBatchTargets = mTestBatchTargets;
%             DPREP_strData.mTestBatchData = mTestBatchData;
%             DPREP_strData.mTestFeatures = mTestFeatures;
%             DPREP_strData.mTestTargets = mTestTargets;
%             DPREP_strData.mTrainFeatures = mTrainFeatures;
%             DPREP_strData.mTrainTargets = mTrainTargets;
            %DPREP_strData.mAutoLabels = mAutoLabels;
			
			% Start the learning process
			LM_startLearningProcessDNN_HMM(CONFIG_strParams,DPREP_strData.mTrainFeatures, DPREP_strData.mTrainTargets, DPREP_strData.mTestFeatures, DPREP_strData.mTestTargets,...
									   DPREP_strData.mTrainBatchTargets, DPREP_strData.mTrainBatchData, DPREP_strData.mTestBatchTargets, DPREP_strData.mTestBatchData,...
                                       DPREP_strData.nBitfieldLength, DPREP_strData.vChunkLength, DPREP_strData.vOffset);
        case 'DNN_SVM'
            
            fprintf(1, 'Train DNN...\n');
            
			fprintf(1, 'Making train and test batches...\n');
			% Make train and test batches
			[DPREP_strData.mTrainBatchTargets, DPREP_strData.mTrainBatchData, DPREP_strData.mTestBatchTargets, DPREP_strData.mTestBatchData] =... 
				BM_makeBatches(DPREP_strData.mTestFeatures,...
							   DPREP_strData.mTestTargets,...
							   DPREP_strData.mTrainFeatures,...
							   DPREP_strData.mTrainTargets,...
							   CONFIG_strParams.nMaxFeaturesRange,...
							   CONFIG_strParams.bAutoLabel,...
							   DPREP_strData.mAutoLabels,...
							   CONFIG_strParams.nBatchSize);
			fprintf(1, 'Batches made successfuly\n');
            
			% Save Batch data
			save(CONFIG_strParams.sInputDataWorkspace, '-struct', 'DPREP_strData', '-append',... 
				'mTrainBatchTargets',... 
				'mTrainBatchData',...
				'mTestBatchTargets',...
				'mTestBatchData');

			if(strcmp(CONFIG_strParams.sMemorySavingMode, 'ON'))
				
				DPREP_strData.mTestFeatures = [];
				DPREP_strData.mTestTargets = [];
				DPREP_strData.mTrainFeatures = [];
				DPREP_strData.mTrainTargets = [];
				DPREP_strData.mAutoLabels = [];
            end
			
%             load 'input_data';
%             DPREP_strData.mTrainBatchTargets = mTrainBatchTargets;
%             DPREP_strData.mTrainBatchData = mTrainBatchData;
%             DPREP_strData.mTestBatchTargets = mTestBatchTargets;
%             DPREP_strData.mTestBatchData = mTestBatchData;
%             DPREP_strData.mTestFeatures = mTestFeatures;
%             DPREP_strData.mTestTargets = mTestTargets;
%             DPREP_strData.mTrainFeatures = mTrainFeatures;
%             DPREP_strData.mTrainTargets = mTrainTargets;
            %DPREP_strData.mAutoLabels = mAutoLabels;
			
			% Start the learning process
			LM_startLearningProcessDNN_HMM(CONFIG_strParams,DPREP_strData.mTrainFeatures, DPREP_strData.mTrainTargets, DPREP_strData.mTestFeatures, DPREP_strData.mTestTargets,...
                                           DPREP_strData.mTrainBatchTargets, DPREP_strData.mTrainBatchData, DPREP_strData.mTestBatchTargets, DPREP_strData.mTestBatchData,...
                                           DPREP_strData.nBitfieldLength, DPREP_strData.vChunkLength, DPREP_strData.vOffset);
        case 'MAXENT'
            
            % Start the learning process
            LM_startLearningProcessMAXENT(CONFIG_strParams,...
                                          DPREP_strData.mTrainFeatures, DPREP_strData.mTrainTargets, DPREP_strData.mTestFeatures, DPREP_strData.mTestTargets);			
            
        case 'AVG'
            LM_startLearningProcessAVG(CONFIG_strParams,...
                                       DPREP_strData.mTrainFeatures, DPREP_strData.mTrainTargets, DPREP_strData.mTestFeatures, DPREP_strData.mTestTargets);
                                   
        case 'NAIVEBAYES'
            LM_startLearningProcessNAIVEBAYES(CONFIG_strParams,...
                                       DPREP_strData.mTrainFeatures, DPREP_strData.mTrainTargets, DPREP_strData.mTestFeatures, DPREP_strData.mTestTargets);
            
        case 'MAXENT_DNN'
            load 'input_data' mTrainFeatures mTrainTargets mTestFeatures mTestTargets; 
            DPREP_strData.mTrainFeatures = mTrainFeatures;
            DPREP_strData.mTrainTargets = mTrainTargets;
            DPREP_strData.mTestFeatures = mTestFeatures;
            DPREP_strData.mTestTargets  = mTestTargets;
            clear mTrainFeatures mTrainTargets mTestFeatures mTestTargets; 
            % Start the learning process of MAXENT. Learning parameters are
            % saved in workspace
            LM_startLearningProcessMAXENT(CONFIG_strParams,...
                                          DPREP_strData.mTrainFeatures, DPREP_strData.mTrainTargets, DPREP_strData.mTestFeatures, DPREP_strData.mTestTargets);
                                      
            % Load MAXENT model params
            load(CONFIG_strParams.sMaxEntWorkSpaceFileName);
            
            % Obtain the probs out of MAXENT as inputs to DNN to fine tune
            DPREP_strData.mTrainFeatures = marginals(MAXENT_clsParams, DPREP_strData.mTrainFeatures, 1);
            DPREP_strData.mTestFeatures = marginals(MAXENT_clsParams, DPREP_strData.mTestFeatures, 1);
            

            % Start the DNN learning process
            fprintf(1, 'Making train and test batches...\n');
			% Make train and test batches
			[DPREP_strData.mTrainBatchTargets, DPREP_strData.mTrainBatchData, DPREP_strData.mTestBatchTargets, DPREP_strData.mTestBatchData] =... 
				BM_makeBatches(DPREP_strData.mTestFeatures,...
							   DPREP_strData.mTestTargets,...
							   DPREP_strData.mTrainFeatures,...
							   DPREP_strData.mTrainTargets,...
							   CONFIG_strParams.nMaxFeaturesRange,...
							   CONFIG_strParams.bAutoLabel,...
							   DPREP_strData.mAutoLabels,...
							   CONFIG_strParams.nBatchSize);
			fprintf(1, 'Batches made successfuly\n');

			% Save Batch data
			save(CONFIG_strParams.sInputDataWorkspace, '-struct', 'DPREP_strData', '-append',... 
				'mTrainBatchTargets',... 
				'mTrainBatchData',...
				'mTestBatchTargets',...
				'mTestBatchData');

			if(strcmp(CONFIG_strParams.sMemorySavingMode, 'ON'))
				
				DPREP_strData.mTestFeatures = [];
				DPREP_strData.mTestTargets = [];
				DPREP_strData.mTrainFeatures = [];
				DPREP_strData.mTrainTargets = [];
				DPREP_strData.mAutoLabels = [];
			end
			
			
			% Start the learning process
			LM_startLearningProcessDNN(CONFIG_strParams,...
                                       DPREP_strData.mTestFeatures, DPREP_strData.mTestTargets, DPREP_strData.mTrainFeatures, DPREP_strData.mTrainTargets,...
                                       DPREP_strData.mTrainBatchTargets, DPREP_strData.mTrainBatchData, DPREP_strData.mTestBatchTargets, DPREP_strData.mTestBatchData);

        case 'TFIDF'                
            % TFIDF training is only possible for bitfield, so convert if Raw
            % features
            if(strcmp(CONFIG_strParams.eFeaturesMode, 'Raw'))                
                fprintf(1, 'Converting train and test features to bitfield...\n');
                [mTrainFeatures] = DCONV_convertRawToBitfield(DPREP_strData.mTrainFeatures, DPREP_strData.nBitfieldLength, DPREP_strData.vChunkLength, DPREP_strData.vOffset);
                [mTestFeatures] = DCONV_convertRawToBitfield(DPREP_strData.mTestFeatures, DPREP_strData.nBitfieldLength, DPREP_strData.vChunkLength, DPREP_strData.vOffset);                
                fprintf(1, 'Converting train and test features to bitfield done successfully\n');
            else
                mTrainFeatures = DPREP_strData.mTrainFeatures;
                mTestFeatures = DPREP_strData.mTestFeatures;
            end
            LM_startLearningProcessTFIDF(CONFIG_strParams,...
                                         mTrainFeatures, DPREP_strData.mTrainTargets, mTestFeatures, DPREP_strData.mTestTargets);

            
	end % end switch
	fprintf(1, 'Learning process performed successfuly\n'); 


	if(CONFIG_strParams.bTestOnIndependentTestSet == 1)
		
		% Open log file
		hFidLog = fopen(CONFIG_strParams.sIndependentDataSetLogFile,'w');
		fprintf(1, 'Testing on independent data set...\n');
		fprintf(hFidLog, 'Testing on independent data set...\n');				
		
		fprintf(1, 'Converting input files...\n');
    
		switch (CONFIG_strParams.sInputFormatOfIndependentTestSet)
			case 'MATLAB'
				% Convert raw data to matlab vars
				[mIndependentDataSetFeatures, mIndependentDataSetTargets] = DCONV_convertMatlabInput_Indepdataset_Binary();
			case 'TxtFile'
				% Convert raw data to matlab vars
				[mIndependentDataSetFeatures, mIndependentDataSetTargets] = DCONV_convert(CONFIG_strParams.sIndependentTestSetFeaturesFilePath);
			otherwise
				% Convert raw data to matlab vars
				[mIndependentDataSetFeatures, mIndependentDataSetTargets] = DCONV_convert(CONFIG_strParams.sIndependentTestSetFeaturesFilePath);
				
		end

		% Save converted data
		save(CONFIG_strParams.sInputDataWorkspace,  '-append', 'mIndependentDataSetFeatures', 'mIndependentDataSetTargets');
		
		fprintf(1, 'Conversion done successfuly\n');
	
		[nErr, nConfusionErr, nErrRate, nConfusionErrRate] = TST_computeClassificationErr(hFidLog, mIndependentDataSetFeatures, mIndependentDataSetTargets, NM_strNetParams, SVM_strParams, CONFIG_strParams);
		
		fprintf(1, 'Testing on independent data set done successfuly\n');
		fprintf(hFidLog, 'Testing on independent data set done successfuly\n');
		
		fclose(hFidLog);
		
		% Save the current configuration in the error performance workspace
		save(CONFIG_strParams.sNameofErrWorkspace, '-append', 'nErr', 'nConfusionErr', 'nErrRate', 'nConfusionErrRate');
	end
end