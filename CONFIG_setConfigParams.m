% Function:
% Sets configuration parameters.
% Inputs:
% None.
% Output:
% CONFIG_strParams: Structure of all configuration parameters
function [CONFIG_strParams] = CONFIG_setConfigParams()

    % Path of the classifier code. For relative path: it'll run from the
    % configuration env. path, so it should be relative to it
    CONFIG_strParams.sDefaultClassifierPath = '..\..\..\..\Generic Classifier\Generic_Classifier_0.16';
    
    % Path to the environment of input and results file of the current
    % configuration. For relative path: it'll run from inside the
    % classifier folder path, so it should be relative to it
	CONFIG_strParams.sConfigEnvPath = pwd;
    
    % Desired reduction of the training set
    % It is represented in the form of percent of the original set size
	CONFIG_strParams.bReduceTrainingSetSizeWithMapping = 1;
    CONFIG_strParams.nDesiredTrainSetSizePercent = 70;
	
	% Flag to be set to 1 if doubling the dataset size is required each mapping phase
	CONFIG_strParams.bDoubleTrainingSetSizeWithMapping = 0;
	
	
    % Configuration of the input format
    % MATLAB: the input is just an auto-generated matlab function setting
    % the matrices values
    % TxtFile: the input is a txt file needs to be parsed
    CONFIG_strParams.sInputFormat = 'MatlabWorkspaceReadyTestTrainSplit';
	
	% Configuration of the dataset to be used
	% MNIST
	% Diacritization
	% TIMIT (Not supported yet)
	CONFIG_strParams.sDataset = 'MNIST';
	
	% The path of the dataset files
	CONFIG_strParams.sDatasetFilesPath = '..\..\Number Recognition\MNIST\Dataset';
	
	% In case subclass training is required, only subset targets are permitted in the dataset
	CONFIG_strParams.vSubClassTargets = 0:9;
    
    % Path of the features file
    CONFIG_strParams.sDefaultPath = CONFIG_strParams.sConfigEnvPath;
    
    % Do u need to save memory: ON or OFF
    CONFIG_strParams.sMemorySavingMode = 'OFF';
    
    % Features file name
    CONFIG_strParams.sFeaturesFileName = 'features_Raw.txt';
    CONFIG_strParams.eFeaturesMode = 'Ready';
    
    % Name of the workspace to save the error structure
    CONFIG_strParams.sNameofErrWorkspace = [CONFIG_strParams.sConfigEnvPath '\err_performance.mat'];
    
    % Name of the input data structures workspace
    CONFIG_strParams.sInputDataWorkspace = [CONFIG_strParams.sConfigEnvPath '\input_data_reduced.mat'];
    
    % Name of the input data structures workspace
    CONFIG_strParams.sNetDataWorkspace = [CONFIG_strParams.sConfigEnvPath '\final_net.mat'];

    % Form the full path of the features file
    CONFIG_strParams.fullRawDataFileName = [CONFIG_strParams.sConfigEnvPath '\' CONFIG_strParams.sFeaturesFileName];
    
    % Split the input data 'uniform' or 'random'
    CONFIG_strParams.sSplitCriteria = 'random';
    
    % Ration of train to test factor ratio
    CONFIG_strParams.nTrainToTestFactor = 10; % Nearly examples are train
    
    % In case of input features not normalized, divide by the max range to
    % normalize
    CONFIG_strParams.nMaxFeaturesRange = 1;
    
    % Batch size to use when making batches (training and testing)
    CONFIG_strParams.nBatchSize = 100;
    
    % Is automating labeling enabled
    CONFIG_strParams.bAutoLabel = 0;
    
    % Name of the workspace storing the automatic labeled data
    CONFIG_strParams.sAutoLabelWorkSpaceFileName = [CONFIG_strParams.sConfigEnvPath '\auto_label.mat'];
    
    % Name of the automatic labels variables in the auto label workspace
    CONFIG_strParams.sAutoLabelNewLabelsVarName = 'new_labels';
    
    % Mapping Direction Modes
    % DEPTH
    % BREADTH
	% SAME
    % NONE
    CONFIG_strParams.eMappingDirection = 'SAME';
    
%     CONFIG_strParams.bDepthMapping = 0;
%     CONFIG_strParams.bBreadthMapping = 1;
    CONFIG_strParams.sLearnLogFile = [CONFIG_strParams.sConfigEnvPath '\learning_log.txt'];
    
    % Number of layers of the initial net, execluding input and top/targets/output layer
    CONFIG_strParams.nInitialNumLayers = 3;% Execluding input and top/targets/output layer
    
    % The architecture of the initial net
    CONFIG_strParams.vInitialLayersWidths = [500 500 2000];
    
    % The final first layer width. This ratio shall be used to inflate all
    % other layers. Example: if init layer width = 100 and final one = 500,
    % then all final layers will be multiplied by 5.
    CONFIG_strParams.nFinalFirstLayerWidth = 1000;
    
    % In case of depth, this is the final depth required.
    CONFIG_strParams.nFinalNumLayers = 3;
    
    % Number of iterations in backprop in which only upper layer weights
    % are updated
    CONFIG_strParams.nBPNumEpochsForUpperLayerTraining = 2;
    
    % Number of epochs in backprop training the basic net before mapping (re-use) starts 
    CONFIG_strParams.nBPNumEpochsBeforeMapping = 200;
    
    % Number of epochs in backprop training during mapping (re-use) phase
    CONFIG_strParams.nBPNumEpochsDuringMapping = 20;
    
    % Iterations to call CG minimizer
    CONFIG_strParams.nMaxIterCGMinimizer = 3;
    
    % For nBPNumEpochsForUpperLayerTraining epochs only those layers will
    % be updated
    CONFIG_strParams.nNumTrainedUpperLayers = 1; % It means update w_class and NW_weights{CONFIG_strParams.nInitialNumLayers} (last layer), so number is the execluding the top layer
    
    % Is pre-training enabled
    CONFIG_strParams.bEnablePretraining = 1;
        if (CONFIG_strParams.bEnablePretraining == 1) 
            % Pre-training (RBM) epochs
            CONFIG_strParams.nPreTrainEpochs = 20;
        else
            CONFIG_strParams.nPreTrainEpochs = 0;
        end % end if
    
    % Flag to indicate if it's desired to update weights in each epoch of
    % backprop only if error is minimized otherwise it's not updated
    CONFIG_strParams.bBPKeepMinWeightsEveryEpoch = 0;   
    
    % Backprop divides each batch into mini batches, each composed of
    % nBPNumExamplesInMiniBatch examples
    CONFIG_strParams.nBPNumExamplesInMiniBatch = 1;
        
    % The following 2 conditions are execlusive    
    % Upper Layer Training Modes during mapping (reuse)
    % TRAIN_UPPER_LAYER_ONLY
    % TRAIN_UPPER_N_LAYERS
    % TRAIN_ALL_LAYERS
    CONFIG_strParams.eBPTrainUpperLayersMode = 'TRAIN_UPPER_LAYER_ONLY';
    
    switch(CONFIG_strParams.eBPTrainUpperLayersMode)
        case 'TRAIN_UPPER_LAYER_ONLY'
            CONFIG_strParams.nBPNumTrainUpperLayerEpochs    = CONFIG_strParams.nBPNumEpochsDuringMapping;
            CONFIG_strParams.nNumTrain_N_UpperLayerEpochs   = 0;

        case 'TRAIN_UPPER_N_LAYERS'
            CONFIG_strParams.nNumTrain_N_UpperLayerEpochs   = CONFIG_strParams.nBPNumEpochsDuringMapping;
            CONFIG_strParams.nBPNumTrainUpperLayerEpochs    = 0;

        case 'TRAIN_ALL_LAYERS'
            CONFIG_strParams.nBPNumTrainUpperLayerEpochs    = CONFIG_strParams.nBPNumEpochsForUpperLayerTraining;
            CONFIG_strParams.nNumTrain_N_UpperLayerEpochs   = 0;

        otherwise
            CONFIG_strParams.nBPNumTrainUpperLayerEpochs    = CONFIG_strParams.nBPNumEpochsForUpperLayerTraining;
            CONFIG_strParams.nNumTrain_N_UpperLayerEpochs   = 0;            
    end % end switch
    
    % Flag to enable penalty methods or not
    CONFIG_strParams.bBPEnablePenaltyIter = 1;
    
        % Barrier Types for penalty method:
        % SQUARE_BARRIER
        % LOG_BARRIER
        CONFIG_strParams.eBarrierType = 'SQUARE_BARRIER';
        
        % The following 2 conditions are execlusive
        %CONFIG_strParams.bLogBarrierPenalty = 0;
        CONFIG_strParams.nLogBarrierMinimizerBeta = 10; % Log barrier parameter

        %CONFIG_strParams.bSquareBarrierPenalty = 1;
        CONFIG_strParams.nSquareBarrierMinimizerAlpha = 10; % Square barrier parameter
        
        % Dynamic barrier means to modify the penalty parameter every
        % iteration
        CONFIG_strParams.bDynamicPenaltyBarrier = 1; % Dynamic update of LOG or SQUARE barriers parameters 
        
        % The dynamic barrier modification (multiplication) parameter
        CONFIG_strParams.nDynMinimizerLambda = 10; % used to multiply the dynamic barrier
        
        % The number of iterations in penalty method
        CONFIG_strParams.nNumPenaltyIterations = 10;

	% Classifier type
	% SVM: Support Vector Machine
	% DNN: Deep Neural Network
	% DNN_HMM: HMM over DNN (input layer to HMM is the target layer output of DNN)
	% AVG: Avergaed classifier
	% MAXENT: Maximum entropy classifier
	% NAIVEBAYES = Naive Bayes
    % MAXENT_DNN: MAXENT then DNN
    % TFIDF: TF-IDF Classifier
	CONFIG_strParams.eClassifierType = 'DNN';
	
	% SVM training kernel function
	% See help svmtrain for possible values (rbf, linear,...etc)
	CONFIG_strParams.eKernelFunction = 'rbf';
    
    % Mapping modes:
    % NN_BREADTH_CLASSIFIER_MAPPING, nn_classifier
    % NN_DEPTH_CLASSIFIER_MAPPING,--> Similar to DEPTH_BASE_UNIT_MAPPING-->
    % -->legacy code CONFIG_NN_depthClassifier
    % WEAK_BREADTH_CLASSIFIER_MAPPING, weak_classifier
    % SEMI_RANDOM_BREADTH_CLASSIFIER_MAPPING, semi_random_mapping
    % FULL_RANDOM_BREADTH_CLASSIFIER_MAPPING, full_random_mapping
    % SVM_TOP_LEVEL_INTEGRATED_CLASSIFIER_MAPPING, svm_top_level_integrated
    % SVM_BREADTH_CLASSIFIER, svm_classifier
    % DEPTH_CASCADED_DATA_REPRESENTATION_MAPPING, CONFIG_depthCascadedDataRepresentation
    % DEPTH_BASE_UNIT_MAPPING, CONFIG_depthBaseUnitMapping 
	% ADAPTIVE: Used for adaptive learning mode. With each phase, the same net architecture is kept but the dataset size is reduced by constant percent = CONFIG_strParams.nDesiredTrainSetSizePercent. The number or mapping phases is detected by CONFIG_strParams.nDesiredAdaptivePhases
	% MappingDirection should be SAME in case of ADAPTIVE
    CONFIG_strParams.eMappingMode = 'ADAPTIVE'; 
	
	% The number or mapping phases is detected by CONFIG_strParams.nDesiredAdaptive mappings
	% Valid only in CONFIG_strParams.eMappingMode = 'ADAPTIVE'
	CONFIG_strParams.nDesiredAdaptivePhases = 2

    % Depth Cascaded Data Representation Modes
    % REPLICATED
    % RANDOMIZE
    CONFIG_strParams.bDepthCascadedDataRepMode = 'REPLICATED';
    
        % The number of units to be stacked over until mapping (reuse)
        % phases are over
        CONFIG_strParams.nDepthBaseUnitMappingNumberOfStackedUnits = 3;
    
    % The SVM trained workspace    
    CONFIG_strParams.sSVMWorkSpaceFileName = [CONFIG_strParams.sConfigEnvPath '\svm_trained.mat'];

    % The HMM trained workspace    
    CONFIG_strParams.sHMMWorkSpaceFileName = [CONFIG_strParams.sConfigEnvPath '\hmm_trained.mat'];
	 
	% The MaxEnt trained workspace 
	CONFIG_strParams.sMaxEntWorkSpaceFileName = [CONFIG_strParams.sConfigEnvPath '\maxent_trained.mat'];
	
	% The Average Classifier trained workspace 
	CONFIG_strParams.sAvgWorkSpaceFileName = [CONFIG_strParams.sConfigEnvPath '\avg_trained.mat'];

	% The Naive Bayes Classifier trained workspace 
	CONFIG_strParams.sNaiveBayesWorkSpaceFileName = [CONFIG_strParams.sConfigEnvPath '\naive_bayes_trained.mat'];
    
    % The TF-IDF Classifier trained workspace
    CONFIG_strParams.sTFIDFWorkSpaceFileName = [CONFIG_strParams.sConfigEnvPath '\tf_idf_trained.mat'];

	% Test on independent data set
	CONFIG_strParams.bTestOnIndependentTestSet = 0;
		
		% Configuration of the input format of the independent set
		% MATLAB: the input is just an auto-generated matlab function setting
		% the matrices values
		% TxtFile: the input is a txt file needs to be parsed
		CONFIG_strParams.sInputFormatOfIndependentTestSet = 'TxtFile';
		
		% Full path to features/targets txt file
	    CONFIG_strParams.sIndependentTestSetFeaturesFilePath  = [CONFIG_strParams.sConfigEnvPath '\features_Binary_Test.txt'];
		
		% Independent DataSet Log File
		CONFIG_strParams.sIndependentDataSetLogFile = [CONFIG_strParams.sConfigEnvPath '\independent_set_log.txt'];
      
	% DNN-HMM context length (order in time instants, or sequence length). This is independent of the context length of the input features.
	CONFIG_strParams.nContextLength = 0;

	% Number of stacked HMM's above DNN-HMM
	CONFIG_strParams.nNumHMMLayers = 1;
	  
	% Test on independent data set. Available in DNN, DNN_HMM, MAXENT
	% For SVM, confusion information is logged in the learning log file
	CONFIG_strParams.bBuildConfusionMatrix = 1;
	
    
end % end function