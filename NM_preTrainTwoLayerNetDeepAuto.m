% Function:
% Run pre-training algorithm to pre-initialize the network. In this case
% the algorithm is RBM greedy learning layer-by-layer
% Inputs:
% mWeigths: The initial 2-layer Net weigthts
% vSecodLayerBiases: The initial upper layer bias
% mVisibleActivations: The input visible layer data
% nPreTrainEpochs: The configured number of epochs
% nSecodLayerWidth: The width of the hidden layer
% NM_strNetParams: The network parameters
% hFidLog: The log file handle
% nBitfieldLength: The bitfield length of the Raw features
% vChunkLength: Vector storing the boundaries of each features chunk
% vOffset: The bitfield boundaries corresponding to each chunk
% eFeaturesMode: Raw, Normal, Binary, Bitfield
% Output:
% mHiddenActivations: The hidden unit activations resulting from pre-training
% mWeigths: The pre-trained weights
% vSecodLayerBiases: The pre-trained hidden layer biases
function [mHiddenActivations, mWeigths, vSecodLayerBiases, vFirstLayerBiases] = NM_preTrainTwoLayerNetDeepAuto(mWeigths, vSecodLayerBiases, mVisibleActivations, nPreTrainEpochs, nSecodLayerWidth, hFidLog, nBitfieldLength, vChunkLength, vOffset, eFeaturesMode, ctrLayer)
        [numcases numdims numbatches]=size(mVisibleActivations);
        
        % Get the sizes of the layer input activation data    
        if(strcmp(eFeaturesMode, 'Raw') & ctrLayer == 1)
            numdims = nBitfieldLength;
        else
            numdims = size(mVisibleActivations, 2);
        end
        numhid = nSecodLayerWidth;
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
        batchdata = mVisibleActivations;
        rbm;
        mHiddenActivations = batchposhidprobs;
        mWeigths = vishid;
        vSecodLayerBiases = hidbiases;
        vFirstLayerBiases = visbiases;
end % end function