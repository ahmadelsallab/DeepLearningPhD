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

	%makebatches_test_train;
	numtargets = size(batchtargets, 2);
   [numcases numdims numbatches]=size(batchdata);
	
	
	%NW_totLayersWidths = [numdims NM_strNetParams.vLayersWidths numtargets];

	if mapping == 1
		load  mnistclassify_weights NW_weights;
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
        clear NW_weights;
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
	end
	LM_strLearningPrcocessPrvt.nPhase=LM_strLearningPrcocessPrvt.nPhase+1;
	test_err_phase(LM_strLearningPrcocessPrvt.nPhase) = min(test_err);
	fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Classification error of LM_strLearningPrcocessPrvt.nPhase %d: %d\n', LM_strLearningPrcocessPrvt.nPhase, min(test_err));
	
end
