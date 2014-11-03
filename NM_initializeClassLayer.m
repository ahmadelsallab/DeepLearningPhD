function [mClassWeights, mPrevClassWeights] = NM_initializeClassLayer(eMappingMode, eMappingDirection, bMapping, sSVMWorkSpaceFileName, mClassWeights, nPhase,...
                                                                      nNumPhases, mTopLayerWeights, nNumTargets) %nTopLayerSize: execluding (before) target layer, including bias%

    % Initialize variables
    mPrevClassWeights = [];
    
    % The nTopLayerSize execluding (before) target layer, including bias
    % (+1)
    nTopLayerSize = size(mTopLayerWeights, 2) + 1;
    
    if(bMapping == 1)
        switch(eMappingDirection)
            case 'BREADTH'
                switch(eMappingMode)
                    case 'SVM_BREADTH_CLASSIFIER'
                        % Load the SVM trained parameters
                        if (nPhase == nNumPhases)
                            load (sSVMWorkSpaceFileName, 'S');
                            %svm_train_test;
                            w_svm = zeros(size(S(1).SupportVectors,2)+1 ,size(S,2));
                            for i = 1:size(S,2)
                                w_svm(:,i)=[(S(i).SupportVectors'*S(i).Alpha); S(i).Bias];
                            end
                        end 
                        
                        % Check if this is the final learning phase
                        if(nPhase == nNumPhases)
                            % Keep the old class weigths. Now, the old weights
                            % are composed of the inflated weigths (X 2) + the
                            % weights for SVM. So the width is 3 times the old
                            % one.
                            mPrevClassWeights = [mClassWeights(1:end-1,:); mClassWeights(1:end-1,:); mClassWeights(end,:)];

                            % Same comment on old weigths above. In addition
                            % the bias is the addition of SVM and NN biases
                            mClassWeights = [mClassWeights(1:end-1,:); mClassWeights(1:end-1,:); w_svm(1:end-1,:); mClassWeights(end,:) + w_svm(end,:)];        
                        else
                            % The new weights
                            % are composed of the inflated weigths (X 2) + the
                            % weights for SVM. So the width is 3 times the old
                            % one.
                            mClassWeights = 0.5*[mClassWeights(1:end-1,:); mClassWeights(1:end-1,:); mClassWeights(end,:)];
                            mPrevClassWeights = mClassWeights;                            
                        end % end if-else
                        
                    % end SVM_BREADTH_CLASSIFIER

                    case 'NN_BREADTH_CLASSIFIER_MAPPING'
                        mClassWeights = 0.5*[mClassWeights(1:end-1,:); mClassWeights(1:end-1,:); mClassWeights(end,:)];
                        mPrevClassWeights = mClassWeights;      
                    % end NN_BREADTH_CLASSIFIER_MAPPING
                    
                    case 'WEAK_BREADTH_CLASSIFIER_MAPPING'
                        mClassWeights = [mClassWeights(1:end-1,:); mClassWeights(end,:); zeros(size(mClassWeights(1:end-1,:),1), size(mClassWeights(1:end-1,:),2)); ];
                        mPrevClassWeights = mClassWeights;                        
                    % end WEAK_BREADTH_CLASSIFIER_MAPPING
                    
                    case 'SEMI_RANDOM_BREADTH_CLASSIFIER_MAPPING'
                        mClassWeights = [mClassWeights(1:end-1,:); mClassWeights(end,:); 0.1*randn(size(mClassWeights(1:end-1,:),1), size(mClassWeights(1:end-1,:),2)); ];
                        mPrevClassWeights = mClassWeights;                        
                    % end SEMI_RANDOM_BREADTH_CLASSIFIER_MAPPING
                    
                    case 'FULL_RANDOM_BREADTH_CLASSIFIER_MAPPING'
                        mClassWeights = [0.1*randn(size(mClassWeights(1:end-1,:),1), size(mClassWeights(1:end-1,:),2)); 0.1*randn(size(mClassWeights,1), size(mClassWeights,2)); ];
                        mPrevClassWeights = mClassWeights;                        
                    % end FULL_RANDOM_BREADTH_CLASSIFIER_MAPPING
                    
                    case 'SVM_TOP_LEVEL_INTEGRATED_CLASSIFIER_MAPPING'
                        mClassWeights = 0.5*[mClassWeights(1:end-1,:); mClassWeights(1:end-1,:); mClassWeights(end,:)];
                        mPrevClassWeights = mClassWeights;                        
                    % end SVM_TOP_LEVEL_INTEGRATED_CLASSIFIER_MAPPING
                    
                end % end switch eMappingMode 'BREADTH'

            case 'DEPTH'
                switch(eMappingMode)
                    case 'DEPTH_CASCADED_DATA_REPRESENTATION_MAPPING'
                        % Normal random initialization just as if no mapping
                        mClassWeights = 0.1*randn(nTopLayerSize, nNumTargets);                        
                        mPrevClassWeights = mClassWeights;                        
                    % end DEPTH_CASCADED_DATA_REPRESENTATION_MAPPING
                    
                    case 'NN_DEPTH_CLASSIFIER_MAPPING'
                        mClassWeights = mClassWeights;
                        mPrevClassWeights = mClassWeights;                        
                    % end NN_DEPTH_CLASSIFIER_MAPPING
                    
                    case 'DEPTH_BASE_UNIT_MAPPING'
                        mPrevClassWeights = mClassWeights;
                        mClassWeights = mClassWeights;                        
                    % end DEPTH_BASE_UNIT_MAPPING
                    
                end % end switch eMappingMode 'DEPTH'
			case 'SAME'
				switch(eMappingMode)
					case 'ADAPTIVE'
                        mClassWeights = mClassWeights;
                        mPrevClassWeights = mClassWeights;  
				end

        end % end switch eMappingDirection
    else
        mClassWeights = 0.1*randn(size(mTopLayerWeights,2)+ 1, nNumTargets);
        
    end % end if-else bMapping
end % end function