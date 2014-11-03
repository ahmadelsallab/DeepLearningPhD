% Version 1.000
%
% Code provided by Ruslan Salakhutdinov and Geoff Hinton  
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our 
% web page. 
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.


% This program pretrains a deep autoencoder for MNIST dataset
% You can set the maximum number of epochs for pretraining each layer
% and you can set the architecture of the multilayer net.

clear all
clc
close all
numhid = 0;
mapping = 0;
CONFIG_strParams.bDepthMapping = 0;
CONFIG_strParams.bBreadthMapping = 1;
LM_strLearningPrcocessPrvt.nPhase = 0;
LM_strLearningPrcocessPrvt.hFidLog = fopen('mapping_phase.txt','w');

% Configurations
CONFIG_strParams.nInitialNumLayers = 3; % Execluding input and top/targets/output layer
%CONFIG_strParams.nInitialLayersWidths = [50 50 100 100 200 400 800];
%CONFIG_strParams.nInitialLayersWidths = [50 100 50 100 50 25];
CONFIG_strParams.nInitialLayersWidths = [125 125 500];
CONFIG_finalFirstLayerWidth = 500;
CONFIG_finalNumLayers = 6;
CONFIG_bpUpperLayerTrainEpochs = 6;
CONFIG_bpNumEpochsBeforeMapping = 50;
CONFIG_bpNumEpochsDuringMapping = 50;
CONFIG_maxIterCGMin = 3;
CONFIG_numTrainUpperLayer = 1; % It means update w_class and NW_weights{CONFIG_strParams.nInitialNumLayers} (last layer), so number is the execluding the top layer
rbm_epochs = 0;
keep_min = 0;
train_upper_layer_only = 1;
train_upper_N_layers_only = 0;
enable_penalty_iter = 1;

	log_barrier_penalty = 0;
    CONFIG_minimizerBeta = 10; % Log barrier parameter
	
    square_barrier_penalty = 1;
    CONFIG_minimizerAlpha = 100; % Square barrier parameter
	
    dynamic_penalty_barrier = 1; % Dynamic update of LOG or SQUARE barriers parameters 
    CONFIG_minimizerLambda = 10; % used to multiply the dynamic barrier
weak_classifier = 0;
semi_random_mapping = 0;
full_random_mapping = 0;
svm_classifier = 0;
nn_classifier = 1;
svm_top_level_integrated = 0;

CONFIG_NN_depthClassifier = 0;
CONFIG_depthCascadedDataRepresentation = 0;
    CONFIG_depthCascadedDataRepReplicated = 0;
    CONFIG_depthCascadedDataRepRandomize = 0;

CONFIG_depthBaseUnitMapping = 0;
CONFIG_depthBaseUnitMappingNumberOfStackedUnits = 3;

CONFIG_bAutoLabel = 1;
CONFIG_bRandFeatures = 1;

% End of Configurations

if CONFIG_strParams.bBreadthMapping == 1
	numphases = floor(CONFIG_finalFirstLayerWidth/CONFIG_strParams.nInitialLayersWidths(1));
	numphases = log(numphases)/log(2);
elseif CONFIG_strParams.bDepthMapping == 1
	if CONFIG_NN_depthClassifier == 1
		numphases = CONFIG_finalNumLayers/CONFIG_strParams.nInitialNumLayers - 1;
		NM_strNetParams.nNumLayers = CONFIG_strParams.nInitialNumLayers;
   elseif CONFIG_depthCascadedDataRepresentation == 1
		numphases = CONFIG_finalNumLayers/CONFIG_strParams.nInitialNumLayers - 1;
		NM_strNetParams.nNumLayers = CONFIG_strParams.nInitialNumLayers;
	elseif CONFIG_depthBaseUnitMapping == 1
		numphases = CONFIG_depthBaseUnitMappingNumberOfStackedUnits - 1;
        NM_strNetParams.nNumLayers = CONFIG_strParams.nInitialNumLayers;
    end
else
    numphases = LM_strLearningPrcocessPrvt.nPhase; 
    NM_strNetParams.nNumLayers = CONFIG_strParams.nInitialNumLayers;
end


while (LM_strLearningPrcocessPrvt.nPhase <= numphases)
	if mapping == 0
        maxepoch = rbm_epochs; 
    else
        maxepoch=0; 
    end
	%if mapping == 0
	%	numhid=numhid_init; numpen=numpen_init; numpen2=numpen2_init; 
	%else
	if(CONFIG_strParams.bBreadthMapping == 1)
		NM_strNetParams.vLayersWidths = CONFIG_strParams.nInitialLayersWidths * 2^LM_strLearningPrcocessPrvt.nPhase;
		NM_strNetParams.nNumLayers = CONFIG_strParams.nInitialNumLayers;
	else
		NM_strNetParams.nPrevNumLayers = NM_strNetParams.nNumLayers;
        NM_strNetParams.vLayersWidths = CONFIG_strParams.nInitialLayersWidths;
		if CONFIG_NN_depthClassifier == 1
			NM_strNetParams.nNumLayers = CONFIG_strParams.nInitialNumLayers * 2^LM_strLearningPrcocessPrvt.nPhase;
			NM_strNetParams.vLayersWidths = [NM_strNetParams.vLayersWidths NM_strNetParams.vLayersWidths];
      elseif CONFIG_depthCascadedDataRepresentation == 1
			% do nothing
		elseif CONFIG_depthBaseUnitMapping == 1
			% do nothing
		end
	end
	fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Training sub-net with arch ');
	for j = 1 : CONFIG_strParams.nInitialNumLayers
		fprintf(LM_strLearningPrcocessPrvt.hFidLog, '%d ', NM_strNetParams.vLayersWidths(j));
	end
	fprintf(LM_strLearningPrcocessPrvt.hFidLog,'\n');

	fprintf(1,'Converting Raw files into Matlab format \n');
	%converter; 

	fprintf(1,'Pretraining a deep autoencoder. \n');
		fprintf(1,'The Science paper used 50 epochs. This uses %3i \n', maxepoch);
	
	if CONFIG_bAutoLabel == 0
		makebatches_test_train;
	else
		makebatches_test_train_auto_label;
	end
	numtargets = size(batchtargets, 2);
   [numcases numdims numbatches]=size(batchdata);
	
	
	%NW_totLayersWidths = [numdims NM_strNetParams.vLayersWidths numtargets];

	if mapping == 1
		load  mnistclassify_weights NW_weights NW_baseUnitWeights;
		save  old_mnistclassify_weights NW_weights;
	end

	if ~(mapping == 1 & CONFIG_depthBaseUnitMapping == 1)
		for layer = 1 : (NM_strNetParams.nNumLayers)
			if	(layer==1)
				fprintf(1,'Pretraining Layer %d with RBM: %d-%d \n', layer, numdims,NM_strNetParams.vLayersWidths(layer));
				fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Pretraining Layer %d with RBM: %d-%d \n', layer, numdims,NM_strNetParams.vLayersWidths(layer));
			else
				fprintf(1,'Pretraining Layer %d with RBM: %d-%d \n', layer, NM_strNetParams.vLayersWidths(layer-1),NM_strNetParams.vLayersWidths(layer));
				fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Pretraining Layer %d with RBM: %d-%d \n', layer, NM_strNetParams.vLayersWidths(layer-1),NM_strNetParams.vLayersWidths(layer));
			end
			restart=1;
			NW;
			rbm;		
			batchdata=batchposhidprobs;		
		end
    else
        clear NW_unitWeights NW_weights;
		NM_strNetParams.nNumLayers = LM_strLearningPrcocessPrvt.nPhase + 1;
		for unit = 1 : NM_strNetParams.nNumLayers
			NW;
		end
	end

	fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Fine tuning ...\n');
    
	backpropclassify_train_testclassifier_mapping; 
    
    if mapping == 1 & CONFIG_strParams.bDepthMapping == 1 & CONFIG_depthCascadedDataRepresentation == 1
            BP_baseUnitWeights = [BP_baseUnitWeights NW_weights];
    end
	%backpropclassify;
	if mapping==0
		mapping = 1;
      BP_baseUnitWeights = NW_weights;
		NW_baseUnitWeights = BP_baseUnitWeights;
        save mnistclassify_weights NW_baseUnitWeights;
	end
	LM_strLearningPrcocessPrvt.nPhase=LM_strLearningPrcocessPrvt.nPhase+1;
	test_err_phase(LM_strLearningPrcocessPrvt.nPhase) = min(test_err);
	fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Classification error of LM_strLearningPrcocessPrvt.nPhase %d: %d\n', LM_strLearningPrcocessPrvt.nPhase, min(test_err));
	
end

plot(test_err_phase);
fclose(LM_strLearningPrcocessPrvt.hFidLog);