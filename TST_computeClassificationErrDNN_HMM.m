% Function:
% Makes feed forward in the net and reports the error.
% Inputs:
% mTrainTargets, mTestTargets: Associated targets. Vector (nxl), where n is the number of examples and l is the number of target classes
% HMM_strParams: HMM transition and emission probs
% Output:
% nTestErr, nTrainErr: number of misclassified examples
function [nTrainErr, nTestErr, vTrainTargetsHMM, vTestTargetsHMM] = TST_computeClassificationErrDNN_HMM(mTrainFeatures, mTrainTargets, mTestFeatures, mTestTargets,...
                                                                     NM_strNetParams, HMM_strParams, CONFIG_strParams)
% The following code is for one shot HMM sequence training and testing
%     % Compute train error
%     % Obtain the output targets of the DNN net                                        
%     [nErr, vTrainTargetDNN] = TST_computeClassificationErrDNN(mTrainFeatures, mTrainTargets, NM_strNetParams, 0, 0, 0, 0, 0, 'ABSOLUTE_ERR_CALC');
% 
%     % Decode corresponding states
%     vTrainTargetsOut = hmmviterbi(vTrainTargetDNN', HMM_strParams.mHMMTransitionProbs, HMM_strParams.mHMMEmissionProbs);
%     
%     % Convert to number = position of 1. Ex: 0010 = 2 (1 @ pos 2)
%     [I J]=max(mTrainTargets, [], 2);
%     vTrainTargets = J';
%     
%     nTrainErr = size(find(vTrainTargets ~= vTrainTargetsOut), 2);
% 
%     % Compute train error
%     % Obtain the output targets of the DNN net                                        
%     [nErr, vTestTargetDNN] = TST_computeClassificationErrDNN(mTestFeatures, mTestTargets, NM_strNetParams, 0, 0, 0, 0, 0, 'ABSOLUTE_ERR_CALC');
% 
%     % Convert to number = position of 1. Ex: 0010 = 2 (1 @ pos 2)
%     [I J]=max(mTestTargets, [], 2);
%     vTestTargets = J';
%     
%     % Decode corresponding states
%     vTestTargetsOut = hmmviterbi(vTestTargetDNN', HMM_strParams.mHMMTransitionProbs, HMM_strParams.mHMMEmissionProbs);
%     nTestErr = size(find(vTestTargets ~= vTestTargetsOut), 2);

% The following code is for context batches HMM training and testing
    %%%%%%%%%%%%%%%%%% TRAINING ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Obtain the output targets of the DNN net                                        
    [nErr, vTrainTargetsDNN] = TST_computeClassificationErrDNN(mTrainFeatures, mTrainTargets, NM_strNetParams, 0, CONFIG_strParams.eMappingDirection,...
                                                               CONFIG_strParams.eMappingMode, 1, 1, 'ABSOLUTE_ERR_CALC');
    % Form the sequences (outputs of DNN)
    vTrainTargetsDNN = vTrainTargetsDNN';

    % The number of states = number of targets
    nStates  = size(mTrainTargets, 2);
    
    % Form the states (required outputs from HMM)
    [I J]=max(mTrainTargets, [], 2);
    vTrainTargets = J';
    
    % Compute train error
    % Initialize the training accumulator error
    nAccTrainErr = 0;
    
    % Get the total number of training examples
    nExamples = size(vTrainTargets, 2);
    
    % Compute context
    if(CONFIG_strParams.nContextLength == 0)
        nContextLength = nExamples;
    else
        nContextLength = CONFIG_strParams.nContextLength;
    end
    
    % Get the integer number of context batches
    N = floor(nExamples/nContextLength);
    
    % Adjust according to the remaining
    if(mod(nExamples, nContextLength) == 0)
        N_HMM = N;
        sizeHMM = nContextLength * ones(1, N_HMM);% sizes of all pieces is equal to context length
    else
        N_HMM = N+1;
        sizeHMM = nContextLength * ones(1, N_HMM);% sizes of all pieces is equal to context length
        sizeHMM(N_HMM) = nExamples - N*nContextLength;% size of last piece is the remaining
    end

    startIdx = 1;
    fprintf(1,'\n');
    i = 1;
    vTargetsHMM = [];
    
    % Start training loop
    while(i <= N_HMM)
        fprintf(1,'Context %d\n', i);
        endIdx = startIdx + sizeHMM(i) - 1;
        
        % The initial targets are the outputs of DNN
        vTargetsDNN = vTrainTargetsDNN(startIdx:endIdx);
        
        % Train number of HMM layers
        for(m = 1 : CONFIG_strParams.nNumHMMLayers)
            
            % Compute the training error
            vTargetsHMM_ = hmmviterbi(vTargetsDNN, HMM_strParams.mHMMTransitionProbs, HMM_strParams.mHMMEmissionProbs);%, 'PSEUDOEMISSIONS', rand(nStates, nStates), 'PSEUDOTRANSITIONS', rand(nStates, nStates));
            
            % The input targets now are the outputs of this layer of HMM
            vTargetsDNN = vTargetsHMM_;
        end % end for
            
        % Accumulate total HMM outputs
        vTargetsHMM = [vTargetsHMM vTargetsHMM_];
        
        % Compute context error
        nTrainErr = size(find(vTargetsHMM_ ~= vTrainTargets(startIdx:endIdx)), 2);
        
        % Accumulate error
        nAccTrainErr = nAccTrainErr + nTrainErr;
        
        % Update the index of the next context
        startIdx = startIdx + sizeHMM(i);
        i = i + 1;

    end % end while

    vTrainTargetsHMM = vTargetsHMM;
    nTrainErr = size(find(vTargetsHMM ~= vTrainTargets), 2);

    %%%%%%%%%%%%%%%%%% TEST ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Obtain the output targets of the DNN net                                        
    [nErr, vTestTargetsDNN] = TST_computeClassificationErrDNN(mTestFeatures, mTestTargets, NM_strNetParams, 0, CONFIG_strParams.eMappingDirection,...
                                                               CONFIG_strParams.eMappingMode, 1, 1, 'ABSOLUTE_ERR_CALC');
    % Form the sequences (outputs of DNN)
    vTestTargetsDNN = vTestTargetsDNN';

    % Form the states (required outputs from HMM)
    [I J]=max(mTestTargets, [], 2);
    vTestTargets = J';    
    
    % Compute train error
    % Initialize the training accumulator error
    nAccTrainErr = 0;
    
    % Get the total number of training examples
    nExamples = size(vTestTargets, 2);
    
    % Compute context
    if(CONFIG_strParams.nContextLength == 0)
        nContextLength = nExamples;
    else
        nContextLength = CONFIG_strParams.nContextLength;
    end
    
    % Get the integer number of context batches
    N = floor(nExamples/nContextLength);
    
    % Adjust according to the remaining
    if(mod(nExamples, nContextLength) == 0)
        N_HMM = N;
        sizeHMM = nContextLength * ones(1, N_HMM);% sizes of all pieces is equal to context length
    else
        N_HMM = N+1;
        sizeHMM = nContextLength * ones(1, N_HMM);% sizes of all pieces is equal to context length
        sizeHMM(N_HMM) = nExamples - N*nContextLength;% size of last piece is the remaining
    end

    startIdx = 1;
    fprintf(1,'\n');
    i = 1;
    vTargetsHMM = [];
    
    % Start training loop
    while(i <= N_HMM)
        fprintf(1,'Context %d\n', i);
        endIdx = startIdx + sizeHMM(i) - 1;
        
        % The initial targets are the outputs of DNN
        vTargetsDNN = vTestTargetsDNN(startIdx:endIdx);
        
        % Train number of HMM layers
        for(m = 1 : CONFIG_strParams.nNumHMMLayers)
            
            % Compute the training error
            vTargetsHMM_ = hmmviterbi(vTargetsDNN, HMM_strParams.mHMMTransitionProbs, HMM_strParams.mHMMEmissionProbs);
            
            % The input targets now are the outputs of this layer of HMM
            vTargetsDNN = vTargetsHMM_;
        end % end for
        
        % Accumulate total HMM outputs
        vTargetsHMM = [vTargetsHMM vTargetsHMM_];
        
        % Compute context error
        nTestErr = size(find(vTargetsHMM_ ~= vTrainTargets(startIdx:endIdx)), 2);
        
        % Accumulate error
        nAccTrainErr = nAccTrainErr + nTestErr;
        
        % Update the index of the next context
        startIdx = startIdx + sizeHMM(i);
        i = i + 1;

    end % end while

    vTestTargetsHMM = vTargetsHMM;
    
    nTestErr = size(find(vTargetsHMM ~= vTestTargets), 2);

end % end function