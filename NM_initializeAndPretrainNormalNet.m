function [NM_strNetParams, mRBMHiddenActivations] = NM_initializeAndPretrainTwoLayerNet(NM_strNetParams, ctrLayer, bMapping, bEnablePretraining, nPreTrainEpochs, mRBMVisibleActivations, hFidLog)
    
    % Get the sizes of the layer input activation data
    [nNumTrainExamples nFirstLayerWidth nNumTrainBatches]=size(mRBMVisibleActivations);
    
    % Get the size of the 
    nSecodLayerWidth = NM_strNetParams.vLayersWidths(ctrLayer);

    if bMapping == 1
        if CONFIG_strParams.bBreadthMapping == 1
            if nn_classifier == 1 | svm_classifier == 1
                if (ctrLayer == 1)
                    mWeights = [NM_strNetParams.cWeights{ctrLayer}(1:end-1,:) NM_strNetParams.cWeights{ctrLayer}(1:end-1,:)];
                else
                    x = NM_strNetParams.cWeights{ctrLayer}(1:end-1,:);
                    mWeights = [NM_strNetParams.cWeights{ctrLayer}(1:end-1,:) zeros(size(x,1), size(x,2)); zeros(size(x,1), size(x,2)) NM_strNetParams.cWeights{ctrLayer}(1:end-1,:)];
                end	
                vSecodLayerBiases  = [NM_strNetParams.cWeights{ctrLayer}(end,:) NM_strNetParams.cWeights{ctrLayer}(end,:)];
                
            elseif semi_random_mapping == 1
                x = NM_strNetParams.cWeights{ctrLayer}(1:end-1,:);
                mWeights     = [NM_strNetParams.cWeights{ctrLayer}(1:end-1,:) 0.1*randn(size(x,1), size(x,2))];
                vSecodLayerBiases  = [NM_strNetParams.cWeights{ctrLayer}(end,:) NM_strNetParams.cWeights{ctrLayer}(end,:)];
                
            elseif full_random_mapping == 1
                x = NM_strNetParams.cWeights{ctrLayer}(1:end-1,:);
                mWeights     = [0.1*randn(size(x,1), size(x,2)) 0.1*randn(size(x,1), size(x,2))];
                vSecodLayerBiases  = [0.1*randn(1,size(x,2)) 0.1*randn(1,size(x,2))];
                
            elseif svm_classifier == 1
                x = NM_strNetParams.cWeights{ctrLayer}(1:end-1,:);
                if (ctrLayer == 1)
                  mWeights = [NM_strNetParams.cWeights{ctrLayer}(1:end-1,:) NM_strNetParams.cWeights{ctrLayer}(1:end-1,:)];
                else
                    mWeights = [NM_strNetParams.cWeights{ctrLayer}(1:end-1,:) zeros(size(x,1), size(x,2)); zeros(size(x,1), size(x,2)) NM_strNetParams.cWeights{ctrLayer}(1:end-1,:)];
                end
                vSecodLayerBiases  = [NM_strNetParams.cWeights{ctrLayer}(end,:) NM_strNetParams.cWeights{ctrLayer}(end,:)];
                
            elseif weak_classifier == 1
                x = NM_strNetParams.cWeights{ctrLayer}(1:end-1,:);
                if (ctrLayer == 1)
                  mWeights = [NM_strNetParams.cWeights{ctrLayer}(1:end-1,:) zeros(size(x,1), size(x,2))];
                else
                    mWeights = [NM_strNetParams.cWeights{ctrLayer}(1:end-1,:) zeros(size(x,1), size(x,2)); zeros(size(x,1), size(x,2)) zeros(size(x,1), size(x,2))];
                end
                vSecodLayerBiases  = [NM_strNetParams.cWeights{ctrLayer}(end,:) zeros(1,size(x,2))];
                
            end
        elseif CONFIG_strParams.bDepthMapping == 1
            if CONFIG_NN_depthClassifier == 1
                if  (ctrLayer == NM_strNetParams.nPrevNumLayers + 1)
                    mWeights     = 0.1*randn(nFirstLayerWidth, nSecodLayerWidth);
                    vSecodLayerBiases  = zeros(1, nSecodLayerWidth);
                    
                else
                    mWeights = NM_strNetParams.cWeights{mod(ctrLayer-1, NM_strNetParams.nPrevNumLayers) + 1}(1:end-1,:);
                    vSecodLayerBiases = NM_strNetParams.cWeights{mod(ctrLayer-1, NM_strNetParams.nPrevNumLayers) + 1}(end,:);
                    
                end
            elseif CONFIG_depthCascadedDataRepresentation == 1
                % Normal random initialization just as if no mapping
                if(ctrLayer == 1)
                    nFirstLayerWidth = size(NM_strNetParams.cWeights{NM_strNetParams.nPrevNumLayers}, 2);
                    mWeights     = 0.1*randn(nFirstLayerWidth, nSecodLayerWidth);                    
                    vSecodLayerBiases  = zeros(1, nSecodLayerWidth);
                else
                        if CONFIG_depthCascadedDataRepReplicated == 1
                    mWeights = NM_strNetParams.cWeights{ctrLayer}(1:end-1,:);
                    vSecodLayerBiases = NM_strNetParams.cWeights{ctrLayer}(end,:);
                        elseif CONFIG_depthCascadedDataRepRandomize == 1
                    mWeights     = 0.1*randn(nFirstLayerWidth, nSecodLayerWidth);                    
                    vSecodLayerBiases  = zeros(1, nSecodLayerWidth);
                  end
                end
            end

            
        end

    else        
        mWeights     = 0.1*randn(nFirstLayerWidth, nSecodLayerWidth);
        vSecodLayerBiases  = zeros(1, nSecodLayerWidth);
        
    end
    
    % Make pre-training if enabled
    if(bEnablePretraining == 1)
        % Adjust the 2-ctrLayer RBM with initialized weights
        vishid = mWeigths;
        hidbiases = vSecodLayerBiases;
        fprintf(1,'Pre-training RBM\n');
        fprintf(hFidLog,'Pre-training RBM\n');
        visbiases  = zeros(1, numdims);
        poshidprobs = zeros(numcases, numhid);
        neghidprobs = zeros(numcases, numhid);
        posprods    = zeros(numdims, numhid);
        negprods    = zeros(numdims, numhid);
        vishidinc  = zeros(numdims, numhid);
        hidbiasinc = zeros(1, numhid);
        visbiasinc = zeros(1, numdims);
        batchposhidprobs=zeros(numcases, numhid ,numbatches);
        maxepoch = nPreTrainEpochs;
        batchdata = mRBMVisibleActivations;
        rbm;
        mRBMHiddenActivations = batchposhidprobs;
    end
    
    mWeigths = vishid;
    vSecodLayerBiases = hidbiases;
    
    NM_strNetParams.cWeights{ctrLayer}=[mWeigths; vSecodLayerBiases];

