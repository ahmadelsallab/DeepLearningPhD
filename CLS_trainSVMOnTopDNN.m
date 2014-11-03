% Function:
% Trains an SVM with the top layer activations of the NN.
% Inputs:
% mTopLayerActivations: The top layer activations of NN
% mTargets: The asscoiated targets
% Output:
% mClassWeights: Updated 1-layer (top NN layer) weights
function [mClassWeights] = CLS_trainSVMOnTopDNN(mTopLayerActivations, mTargets)

    fprintf(1, 'Training SVMs \n');
    % Train nNumTargets SVM's
    nNumTargets = size(mTargets, 2);
    nNumExamples = size(mTargets, 1);

    for j = 1 : nNumTargets
        
        % Convert targets for each SVM
        vSVMTargets = zeros(1, nNumTargets);
        for i = 1 : nNumExamples
            fprintf(1, 'Convert example %d to train SVM %d \n', i, j);
            nSVMTarget = find(mTargets(i,:) == 1);
            if(nSVMTarget == j)
                vSVMTargets(i) = 1;
            else
                vSVMTargets(i) = 0;
            end
        end
        
        % Train SVM's
        
        % If at leat one target is 1 then start SVM training, otherwise SVM
        % training is not possible since there's only 1 class
        if(size(find(vSVMTargets==1), 2) > 0)
            vSVMTargets = vSVMTargets';
            svm_trained(j)=1;
            fprintf(1, 'Start training SVM %d \n', j);
            S(j) = svmtrain(mTopLayerActivations, vSVMTargets, 'kernel_function', 'rbf');
            %S(j) = svmtrain(train_features, vSVMTargets);
            fprintf(1, 'Finished training SVM %d \n', j);
        else
            svm_trained(j)=0;
            fprintf(1, 'SVM %d has no examples in this batch and was not trained\n', j);
        end
    end
    fprintf(1, 'Finished SVMs training\n');
    
    % Map SVM parameters to NN weights (lambda and w0-->weights)
    mSVMClassWeights = zeros(size(mTopLayerActivations,2) + 1 , nNumTargets); % +1 for bias
    
    for i = 1 : nNumTargets
        if svm_trained(i)==1
            mSVMClassWeights(:,i)=[(S(i).SupportVectors'*S(i).Alpha); S(i).Bias];
        else
            % If no 2-class training examples for certain class then set
            % it's weigths to zeros
            mSVMClassWeights(:,i) = zeros(size(mTopLayerActivations,2) + 1, 1);
        end
    end
    % g(x) = w'x + w0
    % w = sum(lambda * yi * xi); yi = +/- 1 according to example class and
    % xi is the features vectors of Support Vectors. sum over Ns: support vectors only.
    
    % Return the NN Class weights
    mClassWeights = mSVMClassWeights;
end % end function