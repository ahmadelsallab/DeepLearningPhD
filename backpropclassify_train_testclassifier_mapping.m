function [NM_strNetParams, TST_strPerformanceInfo] = CLS_fineTuneAndClassifyDNN(NM_strNetParams, CONFIG_strParams, mTrainBatchTargets, mTrainBatchData, mTestBatchTargets, mTestBatchData)

    if svm_classifier == 1 & LM_strLearningPrcocessPrvt.nPhase == numphases
        load enron_svm_kitchen-l.mat S;
        %svm_train_test;
        w_svm = zeros(size(S(1).SupportVectors,2)+1 ,size(S,2));
        for i = 1:size(S,2)
            w_svm(:,i)=[(S(i).SupportVectors'*S(i).Alpha); S(i).Bias];
        end
    end

    % if enable_penalty_iter == 1
         penalty_iter = 10;
    % else
        % penalty_iter = 1;
    % end
    if mapping == 0
        maxepoch = CONFIG_bpNumEpochsBeforeMapping;
    else
        maxepoch = CONFIG_bpNumEpochsDuringMapping;
    end

    if train_upper_layer_only == 1
        %train_upper_layer_epoch = floor(0.75*maxepoch);
        train_upper_layer_epoch = maxepoch;
    else
        train_upper_layer_epoch = CONFIG_bpUpperLayerTrainEpochs;
    end    

    if train_upper_N_layers_only == 1
        %train_upper_layer_epoch = floor(0.75*maxepoch);
        train_upper_N_layers_epoch = maxepoch;
    else
        train_upper_N_layers_epoch = 0;
    end    
    mintesterr = 10000;
    batchsize = 10;
    numminibatches = 1;
    fprintf(1,'\nTraining discriminative model on Enron dataset by minimizing cross entropy error. \n');
    fprintf(1,'9 batches of 10 cases each. \n');

    load mnistvhclassify
    load mnisthpclassify
    load mnisthp2classify

    if CONFIG_bAutoLabel == 0
        makebatches_test_train;
    else
        makebatches_test_train_auto_label;
    end
    load train_test_features_targets.mat test_features test_targets train_features train_targets;
    numtargets = size(train_targets, 2);
    %load train_test_98_features_10_targets_100.mat ;
    [numcases numdims numbatches]=size(batchdata);
    N=numcases; 

    %%%% PREINITIALIZE WEIGHTS OF THE DISCRIMINATIVE MODEL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % if mapping == 1
    % 	load mnistclassify_weights w_class;
    % 	w_class = [w_class(1:end-1,:); w_class(1:end-1,:); w_class(end,:)];
    % else
    % 	w_class = 0.1*randn(size(w3,2)+1,numtargets);
    % end

    % if mapping == 1
    % 	load mnistclassify_weights w_class;
    % 	w_class = [w_class(1:end-1,:); w_class(end,:); zeros(size(w_class(1:end-1,:),1), size(w_class(1:end-1,:),2)); ];
    % else
    % 	w_class = 0.1*randn(size(w3,2)+1,numtargets);
    % end

    if mapping == 1
        load mnistclassify_weights w_class;

        if CONFIG_strParams.bDepthMapping == 1 & CONFIG_depthCascadedDataRepresentation == 1
            BP_prevWeights = BP_baseUnitWeights;
        else
        BP_prevWeights = NW_weights;
        end

        if CONFIG_strParams.bBreadthMapping == 1
            if nn_classifier == 1 | (svm_classifier == 1 & LM_strLearningPrcocessPrvt.nPhase ~= numphases)	
                w_class = 0.5*[w_class(1:end-1,:); w_class(1:end-1,:); w_class(end,:)];
                w_class_prev = w_class;
            elseif weak_classifier == 1
                w_class = [w_class(1:end-1,:); w_class(end,:); zeros(size(w_class(1:end-1,:),1), size(w_class(1:end-1,:),2)); ];
                w_class_prev = w_class;
            elseif semi_random_mapping == 1
                w_class = [w_class(1:end-1,:); w_class(end,:); 0.1*randn(size(w_class(1:end-1,:),1), size(w_class(1:end-1,:),2)); ];
                w_class_prev = w_class;
            elseif full_random_mapping == 1
                w_class = [0.1*randn(size(w_class(1:end-1,:),1), size(w_class(1:end-1,:),2)); 0.1*randn(size(w_class,1), size(w_class,2)); ];
                w_class_prev = w_class;
            elseif svm_classifier == 1 & LM_strLearningPrcocessPrvt.nPhase == numphases
                w_class_prev = [w_class(1:end-1,:); w_class(1:end-1,:); w_class(end,:)];
                w_class = [w_class(1:end-1,:); w_class(1:end-1,:); w_svm(1:end-1,:); w_class(end,:)+w_svm(end,:)];
            elseif svm_top_level_integrated == 1
              w_class = 0.5*[w_class(1:end-1,:); w_class(1:end-1,:); w_class(end,:)];
                w_class_prev = w_class;
            end
        elseif CONFIG_strParams.bDepthMapping == 1
            if CONFIG_NN_depthClassifier == 1
                w_class = w_class;
                w_class_prev = w_class;
            elseif CONFIG_depthCascadedDataRepresentation == 1
                % Normal random initialization just as if no mapping
                w_class = 0.1*randn(size(NW_weights{NM_strNetParams.nNumLayers},2)+1,numtargets);
             w_class_prev = w_class;
            elseif CONFIG_depthBaseUnitMapping == 1
                w_class_prev = w_class;
                w_class = w_class;
            end
        end
    else
        w_class = 0.1*randn(size(NW_weights{NM_strNetParams.nNumLayers},2)+1,numtargets);
    end

    for(layer = 1 : NM_strNetParams.nNumLayers)
        l(layer) = size(NW_weights{layer}, 1) - 1;
    end
    l(NM_strNetParams.nNumLayers+1) = size(w_class, 1)-1;
    l(NM_strNetParams.nNumLayers+2) = numtargets;

    test_err=[];
    train_err=[];

    %%%%%%%%%% END OF PREINITIALIZATIO OF WEIGHTS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    for epoch = 1:maxepoch
    if mapping == 1 & keep_min == 1
        for(layer = 1 : NM_strNetParams.nNumLayers)
            BP_prevWeights{layer} = BP_minWeights{layer};
        end
        w_class_prev = w_class_min;
    end
    %%%%%%%%%%%%%%%%%%%% COMPUTE TRAINING MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    err=0; 
    err_cr=0;
    counter=0;
    [numcases numdims numbatches]=size(batchdata);
    N=numcases;
     for batch = 1:numbatches
      data = [batchdata(:,:,batch)];
      target = [batchtargets(:,:,batch)];
      BP_layerInputData = data;
        if ~(mapping == 1 & CONFIG_depthBaseUnitMapping == 1)
            if mapping == 1 & CONFIG_strParams.bDepthMapping == 1 & CONFIG_depthCascadedDataRepresentation == 1
                [tempDataNotAugmented data] = NM_baseUnitActivation(data, BP_baseUnitWeights);
            else
                data = [data ones(N,1)];
            end

            BP_layerInputData = data;
            for(layer = 1 : NM_strNetParams.nNumLayers)
                BP_wprobs{layer} = 1./(1 + exp(-BP_layerInputData*NW_weights{layer}));
                BP_wprobs{layer} = [BP_wprobs{layer} ones(N,1)];
                BP_layerInputData = [];
                BP_layerInputData = BP_wprobs{layer};
            end
        else
            [tempActivationNonAugmented BP_wprobs] = NM_compositeNetActivation(data, NW_unitWeights);
        end
      if mapping == 0
        targetout = exp(BP_wprobs{NM_strNetParams.nNumLayers}*w_class);
      else
        if svm_classifier == 1 & LM_strLearningPrcocessPrvt.nPhase == numphases
            BP_wprobs{NM_strNetParams.nNumLayers} = [BP_wprobs{NM_strNetParams.nNumLayers}(:,1:end-1) batchdata(:,:,batch) ones(N,1)];
        else
            if CONFIG_strParams.bBreadthMapping == 1
                targetout = 0.5*exp(BP_wprobs{NM_strNetParams.nNumLayers}*w_class);
            elseif CONFIG_strParams.bDepthMapping == 1
                targetout = exp(BP_wprobs{NM_strNetParams.nNumLayers}*w_class);
              elseif CONFIG_depthBaseUnitMapping == 1
                    targetout = exp(BP_wprobs{NM_strNetParams.nNumLayers}*w_class);
            end
        end
      end
      targetout = targetout./repmat(sum(targetout,2),1,numtargets);

      [I J]=max(targetout,[],2);
      [I1 J1]=max(target,[],2);
      counter=counter+length(find(J==J1));
      err_cr = err_cr- sum(sum( target(:,1:end).*log(targetout))) ;
     end
     train_err(epoch)=(numcases*numbatches-counter);
     train_crerr(epoch)=err_cr/numbatches;

    %%%%%%%%%%%%%% END OF COMPUTING TRAINING MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%% COMPUTE TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    err=0;
    err_cr=0;
    counter=0;
    [testnumcases testnumdims testnumbatches]=size(testbatchdata);
    N=testnumcases;
    for batch = 1:testnumbatches
      data = [testbatchdata(:,:,batch)];
      target = [testbatchtargets(:,:,batch)];

        if ~(mapping == 1 & CONFIG_depthBaseUnitMapping == 1)
            if mapping == 1 & CONFIG_strParams.bDepthMapping == 1 & CONFIG_depthCascadedDataRepresentation == 1
             [tempDataNotAugmented data] = NM_baseUnitActivation(data, BP_baseUnitWeights);
            else
             data = [data ones(N,1)];
            end

            BP_layerInputData = data;

            for(layer = 1 : NM_strNetParams.nNumLayers)
                BP_wprobs{layer} = 1./(1 + exp(-BP_layerInputData*NW_weights{layer}));
                BP_wprobs{layer} = [BP_wprobs{layer} ones(N,1)];
                BP_layerInputData = [];
                BP_layerInputData = BP_wprobs{layer};
            end
        else
            [tempActivationNonAugmented BP_wprobs] = NM_compositeNetActivation(data, NW_unitWeights);
        end
      if mapping == 0
        targetout = exp(BP_wprobs{NM_strNetParams.nNumLayers}*w_class);
      else
        if svm_classifier == 1 & LM_strLearningPrcocessPrvt.nPhase == numphases
            BP_wprobs{NM_strNetParams.nNumLayers} = [BP_wprobs{NM_strNetParams.nNumLayers}(:,1:end-1) batchdata(:,:,batch) ones(N,1)];
        else
            if CONFIG_strParams.bBreadthMapping == 1
                targetout = 0.5*exp(BP_wprobs{NM_strNetParams.nNumLayers}*w_class);
            elseif CONFIG_strParams.bDepthMapping == 1
                targetout = exp(BP_wprobs{NM_strNetParams.nNumLayers}*w_class);
            elseif CONFIG_depthBaseUnitMapping == 1
                targetout = exp(BP_wprobs{NM_strNetParams.nNumLayers}*w_class);
            end
        end
      end
      targetout = targetout./repmat(sum(targetout,2),1,numtargets);

      [I J]=max(targetout,[],2);
      [I1 J1]=max(target,[],2);
      counter=counter+length(find(J==J1));
      err_cr = err_cr - sum(sum( target(:,1:end).*log(targetout))) ;
    end
     test_err(epoch)=(testnumcases*testnumbatches-counter);
     test_crerr(epoch)=err_cr/testnumbatches;
     fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Before epoch %d Train # misclassified: %d (from %d). Test # misclassified: %d (from %d) \t \t \n',...
                epoch,train_err(epoch),numcases*numbatches,test_err(epoch),testnumcases*testnumbatches);
     fprintf(1,'Before epoch %d Train # misclassified: %d (from %d). Test # misclassified: %d (from %d) \t \t \n',...
                epoch,train_err(epoch),numcases*numbatches,test_err(epoch),testnumcases*testnumbatches);
     if test_err(epoch) <= mintesterr
        mintesterr = test_err(epoch);
        for(layer = 1 : NM_strNetParams.nNumLayers)
            BP_minWeights{layer} = NW_weights{layer};
        end
        w_class_min = w_class;
     end

     %%%%%%%%%%%%%% END OF COMPUTING TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

     tt=0;
     for batch = 1:numbatches/numminibatches
     fprintf(1,'epoch %d batch %d\r',epoch,batch);

    %%%%%%%%%%% COMBINE numminibatches MINIBATCHES INTO 1 LARGER MINIBATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     tt=tt+1; 
     data=[];
     targets=[]; 
     for kk=1:numminibatches
      data=[data 
            batchdata(:,:,(tt-1)*numminibatches+kk)]; 
      targets=[targets
            batchtargets(:,:,(tt-1)*numminibatches+kk)];
     end 

    %%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%


      if mapping == 0
        if epoch < train_upper_layer_epoch  % First update top-level weights holding other weights fixed. 
          N = size(data,1);
          XX = [data ones(N,1)];
            BP_layerInputData = XX;
            for(layer = 1 : NM_strNetParams.nNumLayers)
                BP_wprobs{layer} = 1./(1 + exp(-BP_layerInputData*NW_weights{layer}));
                BP_wprobs{layer} = [BP_wprobs{layer} ones(N,1)];
                BP_layerInputData = [];
                BP_layerInputData = BP_wprobs{layer};
            end
            BP_wprobs{NM_strNetParams.nNumLayers} = BP_wprobs{NM_strNetParams.nNumLayers}(:,1:end-1);

            VV = [w_class(:)']';
            Dim = [l(NM_strNetParams.nNumLayers+1); l(NM_strNetParams.nNumLayers+2)];
            if svm_top_level_integrated == 1
                fprintf(1, 'Training SVMs \n');
                % Train num_targets SVM's

                for j = 1 : size(targets, 2)
                    labels=zeros(1, size(targets, 2));
                    for i = 1 : size(targets, 1)
                        fprintf(1, 'Convert example %d to train SVM %d \n', i, j);
                        target_svm = find(targets(i,:) == 1);
                        if(target_svm == j)
                            labels(i) = 1;
                        else
                            labels(i) = 0;
                        end
                    end
                     if(size(find(labels==1),2)>0)
                        labels = labels';
                        svm_trained(j)=1;
                        fprintf(1, 'Start training SVM %d \n', j);
                        S(j) = svmtrain(BP_wprobs{NM_strNetParams.nNumLayers}, labels, 'kernel_function', 'rbf');
                        %S(j) = svmtrain(train_features, labels);
                        fprintf(1, 'Finished training SVM %d \n', j);
                    else
                        svm_trained(j)=0;
                        fprintf(1, 'SVM %d has no examples in this batch and was not trained\n', j);
                    end


                end
                fprintf(1, 'Finished SVMs training\n');
                w_svm = zeros(size(BP_wprobs{NM_strNetParams.nNumLayers},2)+1 ,size(targets,2));
                for i = 1:size(targets,2)
                    if svm_trained(i)==1
                        w_svm(:,i)=[(S(i).SupportVectors'*S(i).Alpha); S(i).Bias];
                    else
                        w_svm(:,i) = zeros(size(BP_wprobs{NM_strNetParams.nNumLayers},2)+1,1);
                    end
                end
                w_class = w_svm;
            else
                [X, fX] = minimize(VV,'CG_CLASSIFY_INIT',CONFIG_maxIterCGMin,Dim,BP_wprobs{NM_strNetParams.nNumLayers},targets);
                w_class = reshape(X,l(NM_strNetParams.nNumLayers+1)+1,l(NM_strNetParams.nNumLayers+2));
            end

        else
            VV = [];
            for(layer = 1 : NM_strNetParams.nNumLayers)
                VV = [VV NW_weights{layer}(:)'];
            end
            VV = [VV w_class(:)'];
            VV = VV';

            Dim = [];
            for(layer = 1 : (NM_strNetParams.nNumLayers+2))
                Dim = [Dim; l(layer)];
            end

            [X, fX] = minimize(VV,'CG_CLASSIFY',CONFIG_maxIterCGMin,Dim,data,targets, 0, 0, 0, 0);

            offset = 0;
            for(layer = 1 : NM_strNetParams.nNumLayers)
                NW_weights{layer} = reshape(X(offset+1:offset+(l(layer)+1)*l(layer+1)), l(layer)+1, l(layer+1));
                offset = offset + (l(layer)+1)*l(layer+1);
            end

            if svm_top_level_integrated == 1
                N = size(data,1);
                XX = [data ones(N,1)];
                BP_layerInputData = XX;
                for(layer = 1 : NM_strNetParams.nNumLayers)
                    BP_wprobs{layer} = 1./(1 + exp(-BP_layerInputData*NW_weights{layer}));
                    BP_wprobs{layer} = [BP_wprobs{layer} ones(N,1)];
                    BP_layerInputData = [];
                    BP_layerInputData = BP_wprobs{layer};
                end
                BP_wprobs{NM_strNetParams.nNumLayers} = BP_wprobs{NM_strNetParams.nNumLayers}(:,1:end-1);

                fprintf(1, 'Training SVMs \n');

                % Train num_targets SVM's
                for j = 1 : size(targets, 2)
                    labels=zeros(1, size(targets, 2));
                    for i = 1 : size(targets, 1)
                        fprintf(1, 'Convert example %d to train SVM %d \n', i, j);
                        target_svm = find(targets(i,:) == 1);
                        if(target_svm == j)
                            labels(i) = 1;
                        else
                            labels(i) = 0;
                        end
                    end
                    if(size(find(labels==1),2)>0)
                        labels = labels';
                        svm_trained(j)=1;
                        fprintf(1, 'Start training SVM %d \n', j);
                        S(j) = svmtrain(BP_wprobs{NM_strNetParams.nNumLayers}, labels, 'kernel_function', 'rbf');
                        %S(j) = svmtrain(train_features, labels);
                        fprintf(1, 'Finished training SVM %d \n', j);
                    else
                        svm_trained(j)=0;
                        fprintf(1, 'SVM %d has no examples in this batch and was not trained\n', j);
                    end


                end
                fprintf(1, 'Finished SVMs training\n');
                w_svm = zeros(size(BP_wprobs{NM_strNetParams.nNumLayers},2)+1 ,size(targets,2));
                for i = 1:size(targets,2)
                    if svm_trained(i)==1
                        w_svm(:,i)=[(S(i).SupportVectors'*S(i).Alpha); S(i).Bias];
                    else
                        w_svm(:,i) = zeros(size(BP_wprobs{NM_strNetParams.nNumLayers},2)+1,1);
                    end
                end
                w_class = w_svm;
            else
                w_class = reshape(X(offset+1:offset+(l(NM_strNetParams.nNumLayers+1)+1)*l(NM_strNetParams.nNumLayers+2)),l(NM_strNetParams.nNumLayers+1)+1,l(NM_strNetParams.nNumLayers+2));
            end

        end
      else
        if epoch <= train_upper_layer_epoch  % First update top-level weights holding other weights fixed. 

            if ~(mapping == 1 & CONFIG_depthBaseUnitMapping == 1)
                N = size(data,1);

                if CONFIG_strParams.bDepthMapping == 1 & CONFIG_depthCascadedDataRepresentation == 1
                 [tempDataNotAugmented XX] = NM_baseUnitActivation(data, BP_baseUnitWeights);
                else
                 XX = [data ones(N,1)];
                end


                BP_layerInputData = XX;
                for(layer = 1 : NM_strNetParams.nNumLayers)
                    BP_wprobs{layer} = 1./(1 + exp(-BP_layerInputData*NW_weights{layer}));
                    BP_wprobs{layer} = [BP_wprobs{layer} ones(N,1)];
                    BP_layerInputData = [];
                    BP_layerInputData = BP_wprobs{layer};
                end
                BP_wprobs{NM_strNetParams.nNumLayers} = BP_wprobs{NM_strNetParams.nNumLayers}(:,1:end-1);

                if CONFIG_strParams.bBreadthMapping == 1 & svm_classifier == 1 & LM_strLearningPrcocessPrvt.nPhase == numphases
                    BP_wprobs{NM_strNetParams.nNumLayers} = [BP_wprobs{NM_strNetParams.nNumLayers} data];
                end
            else
                [BP_wprobs tempActivationAugmented] = NM_compositeNetActivation(data, NW_unitWeights);
            end
            if enable_penalty_iter == 1
                for i = 1 : penalty_iter				
                    VV = [w_class(:)']';
                    Dim = [l(NM_strNetParams.nNumLayers+1); l(NM_strNetParams.nNumLayers+2)];
                    if ~(mapping == 1 & CONFIG_depthBaseUnitMapping == 1)
                        BP_layerInputData = [data ones(N,1)];
                        for(layer = 1 : NM_strNetParams.nNumLayers)
                            BP_prevWprobs{layer} = 1./(1 + exp(-BP_layerInputData*BP_prevWeights{layer}));
                            BP_prevWprobs{layer} = [BP_prevWprobs{layer} ones(N,1)];
                            BP_layerInputData = [];
                            BP_layerInputData = BP_prevWprobs{layer};
                        end
                        BP_prevWprobs{NM_strNetParams.nNumLayers} = BP_prevWprobs{NM_strNetParams.nNumLayers}(:,1:end-1);
                    else
                        [BP_prevWprobs tempActivationNonAugmented] = NM_neuralNetActivation(data, NW_baseUnitWeights);
                    end

                    if (log_barrier_penalty == 1)

                        beta = CONFIG_minimizerBeta;
                        lambda = CONFIG_minimizerLambda;
                        [X, fX] = minimize(VV,'CG_CLASSIFY_INIT_CONSTRAINED_LOG',CONFIG_maxIterCGMin,Dim,BP_wprobs{NM_strNetParams.nNumLayers},targets, BP_prevWprobs{end}, w_class_prev, beta);
                        w_class = reshape(X,l(NM_strNetParams.nNumLayers+1)+1,l(NM_strNetParams.nNumLayers+2));

                        % dynamic beta
                        if (dynamic_penalty_barrier == 1)
                            beta = lambda*beta;
                        end

                    elseif (square_barrier_penalty == 1)
                        lambda = CONFIG_minimizerLambda;
                        alpha = CONFIG_minimizerAlpha;
                        [X, fX] = minimize(VV,'CG_CLASSIFY_INIT_CONSTRAINED_SQUARE',CONFIG_maxIterCGMin,Dim,BP_wprobs{NM_strNetParams.nNumLayers},targets, BP_prevWprobs{end}, w_class_prev, alpha);
                        w_class = reshape(X,l(NM_strNetParams.nNumLayers+1)+1,l(NM_strNetParams.nNumLayers+2));
                        if (dynamic_penalty_barrier == 1)
                            alpha = alpha*lambda;
                        end
                    end
                end
            else
                VV = [w_class(:)']';
                Dim = [l(NM_strNetParams.nNumLayers+1); l(NM_strNetParams.nNumLayers+2)];
                [X, fX] = minimize(VV,'CG_CLASSIFY_INIT',CONFIG_maxIterCGMin,Dim,BP_wprobs{NM_strNetParams.nNumLayers},targets);
             w_class = reshape(X,l(NM_strNetParams.nNumLayers+1)+1,l(NM_strNetParams.nNumLayers+2));
            end
        else
            if epoch < train_upper_N_layers_epoch
                if ~(mapping == 1 & CONFIG_depthBaseUnitMapping == 1)
                    N = size(data,1);
                    if CONFIG_strParams.bDepthMapping == 1 & CONFIG_depthCascadedDataRepresentation == 1
                         [tempDataNotAugmented XX] = NM_baseUnitActivation(data, BP_baseUnitWeights);
                    else
                         XX = [data ones(N,1)];
                    end

                    BP_layerInputData = XX;
                    for(layer = 1 : (NM_strNetParams.nNumLayers - CONFIG_numTrainUpperLayer))
                        BP_wprobs{layer} = 1./(1 + exp(-BP_layerInputData*NW_weights{layer}));
                        BP_wprobs{layer} = [BP_wprobs{layer} ones(N,1)];
                        BP_layerInputData = [];
                        BP_layerInputData = BP_wprobs{layer};
                    end
                    BP_wprobs{layer} = BP_wprobs{layer}(:, 1:end-1);
                else
                    [BP_wprobs tempActivationAugmented] = NM_compositeNetActivation(data, NW_unitWeights);
                end

                if enable_penalty_iter == 1
                    for i = 1 : penalty_iter
                        VV = [];
                        Dim = [];
                        for layer = ((NM_strNetParams.nNumLayers - CONFIG_numTrainUpperLayer)+1) : NM_strNetParams.nNumLayers
                            VV = [VV NW_weights{layer}(:)'];
                            Dim = [Dim; l(layer)];
                        end
                        VV = [VV w_class(:)']';
                        Dim = [Dim; l(NM_strNetParams.nNumLayers+1); l(NM_strNetParams.nNumLayers+2)];
                        if ~(mapping == 1 & CONFIG_depthBaseUnitMapping == 1)
                            BP_layerInputData = [data ones(N,1)];
                            for(layer = 1 : (NM_strNetParams.nNumLayers-CONFIG_numTrainUpperLayer))
                                BP_prevWprobs{layer} = 1./(1 + exp(-BP_layerInputData*BP_prevWeights{layer}));
                                BP_prevWprobs{layer} = [BP_prevWprobs{layer} ones(N,1)];
                                BP_layerInputData = [];
                                BP_layerInputData = BP_prevWprobs{layer};
                            end
                            BP_prevWprobs{layer} = BP_prevWprobs{layer}(:,1:end-1);
                        else
                            [BP_prevWprobs tempActivationAugmented] = NM_neuralNetActivation(data, NW_baseUnitWeights);
                        end

                        alpha = CONFIG_minimizerAlpha;
                        lambda = CONFIG_minimizerLambda;

                        if (CONFIG_depthBaseUnitMapping == 0)
                            [X, fX] = minimize(VV,'CG_CLASSIFY_N_Layers_CONSTRAINED',CONFIG_maxIterCGMin,Dim,BP_wprobs{NM_strNetParams.nNumLayers-CONFIG_numTrainUpperLayer},targets, BP_prevWprobs{end-CONFIG_numTrainUpperLayer}, BP_prevWeights, w_class_prev, alpha, CONFIG_depthBaseUnitMapping, 0, 0, 0);
                        else
                            [X, fX] = minimize(VV,'CG_CLASSIFY_N_Layers_CONSTRAINED',CONFIG_maxIterCGMin,Dim,BP_wprobs{NM_strNetParams.nNumLayers-CONFIG_numTrainUpperLayer},targets, BP_prevWprobs{end-CONFIG_numTrainUpperLayer}, NW_baseUnitWeights, w_class_prev, alpha, CONFIG_depthBaseUnitMapping, NW_unitWeights{NM_strNetParams.nNumLayers-CONFIG_numTrainUpperLayer}, NW_weights{NM_strNetParams.nNumLayers-CONFIG_numTrainUpperLayer}, w_class);
                        end
                        if (dynamic_penalty_barrier == 1)
                            alpha = alpha*lambda;
                        end

                        offset = 0;
                        for layer = ((NM_strNetParams.nNumLayers-CONFIG_numTrainUpperLayer)+1) : NM_strNetParams.nNumLayers
                            NW_weights{layer} = reshape(X(offset + 1 : offset + size(NW_weights{layer}(:)',2)), size(NW_weights{layer} , 1), size(NW_weights{layer} , 2));
                            offset = offset + size(NW_weights{layer}(:)',2);
                        end
                        w_class = reshape(X(offset + 1 : offset+size(w_class(:)',2)), size(w_class , 1), size(w_class , 2));

                        % Update base units input layer weights by intermediate weights
                        for tempUnit = ((NM_strNetParams.nNumLayers-CONFIG_numTrainUpperLayer)+1) : NM_strNetParams.nNumLayers
                            NW_unitWeights{tempUnit}{1} = NW_weights{tempUnit};
                        end

                    end
                else
                    VV = [];
                    Dim = [];
                    for layer = ((NM_strNetParams.nNumLayers-CONFIG_numTrainUpperLayer)+1) : NM_strNetParams.nNumLayers
                        VV = [VV NW_weights{layer}(:)'];
                        Dim = [Dim; l(layer)];
                    end
                    VV = [VV w_class(:)']';
                    Dim = [Dim; l(NM_strNetParams.nNumLayers+1); l(NM_strNetParams.nNumLayers+2)];

                    [X, fX] = minimize(VV,'CG_CLASSIFY_N_Layers',CONFIG_maxIterCGMin,Dim,BP_wprobs{NM_strNetParams.nNumLayers-CONFIG_numTrainUpperLayer},targets, CONFIG_depthBaseUnitMapping, NW_unitWeights{NM_strNetParams.nNumLayers-CONFIG_numTrainUpperLayer}, NW_weights{NM_strNetParams.nNumLayers-CONFIG_numTrainUpperLayer}, w_class);

                    offset = 0;
                    for layer = ((NM_strNetParams.nNumLayers-CONFIG_numTrainUpperLayer)+1) : NM_strNetParams.nNumLayers
                        NW_weights{layer} = reshape(X(offset + 1 : offset + size(NW_weights{layer}(:)',2)), size(NW_weights{layer} , 1), size(NW_weights{layer} , 2));
                        offset = offset + size(NW_weights{layer}(:)',2);
                    end
                    w_class = reshape(X(offset + 1 : offset+size(w_class(:)',2)), size(w_class , 1), size(w_class , 2));

                    % Update base units input layer weights by intermediate weights
                    for tempUnit = ((NM_strNetParams.nNumLayers-CONFIG_numTrainUpperLayer)+1) : NM_strNetParams.nNumLayers
                        NW_unitWeights{tempUnit}{1} = NW_weights{tempUnit};
                    end
                end
            else
                if enable_penalty_iter == 1
                    for i = 1 : penalty_iter				
                        VV = [];
                        for(layer = 1 : NM_strNetParams.nNumLayers)
                            VV = [VV NW_weights{layer}(:)'];
                        end
                        VV = [VV w_class(:)'];
                        VV = VV';

                        Dim = [];
                        for(layer = 1 : (NM_strNetParams.nNumLayers+2))
                            Dim = [Dim; l(layer)];
                        end


                        alpha = CONFIG_minimizerAlpha;
                        lambda = CONFIG_minimizerLambda;
                        if CONFIG_depthBaseUnitMapping == 1
                            [X, fX] = minimize(VV,'CG_CLASSIFY_CONSTRAINED',CONFIG_maxIterCGMin,Dim,data,targets, NW_baseUnitWeights, data, w_class_prev, alpha, CONFIG_depthBaseUnitMapping, NW_unitWeights, NW_weights, w_class);
                        else
                            [X, fX] = minimize(VV,'CG_CLASSIFY_CONSTRAINED',CONFIG_maxIterCGMin,Dim,data,targets, BP_prevWeights, data, w_class_prev, alpha, CONFIG_depthBaseUnitMapping, NW_unitWeights, NW_weights, w_class);
                        end

                        if (dynamic_penalty_barrier == 1)
                            alpha = alpha*lambda;
                        end

                        offset = 0;
                        for layer = 1 : NM_strNetParams.nNumLayers
                            NW_weights{layer} = reshape(X(offset + 1 : offset + size(NW_weights{layer}(:)',2)), size(NW_weights{layer} , 1), size(NW_weights{layer} , 2));
                            offset = offset + size(NW_weights{layer}(:)',2);
                        end
                        w_class = reshape(X(offset + 1 : offset+size(w_class(:)',2)), size(w_class , 1), size(w_class , 2));

                        % Update base units input layer weights by intermediate weights
                        for tempUnit = 1 : size(NW_weights ,2)
                            NW_unitWeights{tempUnit}{1} = NW_weights{tempUnit};
                        end
                    end
                else
                    VV = [];
                    for(layer = 1 : NM_strNetParams.nNumLayers)
                        VV = [VV NW_weights{layer}(:)'];
                    end
                    VV = [VV w_class(:)'];
                    VV = VV';

                    Dim = [];
                    for(layer = 1 : (NM_strNetParams.nNumLayers+2))
                        Dim = [Dim; l(layer)];
                    end

                    if CONFIG_strParams.bDepthMapping == 1 & CONFIG_depthCascadedDataRepresentation == 1
                      [XX tempDataAugmented] = NM_baseUnitActivation(data, BP_baseUnitWeights);
                    else
                      %XX = [data ones(N,1)];
                    end

                    [X, fX] = minimize(VV,'CG_CLASSIFY',CONFIG_maxIterCGMin,Dim,data,targets, CONFIG_depthBaseUnitMapping, NW_unitWeights, NW_weights, w_class);

                    offset = 0;
                    for layer = 1 : NM_strNetParams.nNumLayers
                        NW_weights{layer} = reshape(X(offset + 1 : offset + size(NW_weights{layer}(:)',2)), size(NW_weights{layer} , 1), size(NW_weights{layer} , 2));
                        offset = offset + size(NW_weights{layer}(:)',2);
                    end
                    w_class = reshape(X(offset + 1 : offset+size(w_class(:)',2)), size(w_class , 1), size(w_class , 2));

                    % Update base units input layer weights by intermediate weights
                    for tempUnit = 1 : size(NW_weights ,2)
                        NW_unitWeights{tempUnit}{1} = NW_weights{tempUnit};
                    end
                end
            end
        end 
       end
    %%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

     end
    % for batch = 1:testnumbatches
      % data = [testbatchdata(:,:,batch)];
      % target = [testbatchtargets(:,:,batch)];
      % data = [data ones(N,1)];
      % w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)];
      % w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
      % w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
      % if mapping == 0
        % targetout = exp(w3probs*w_class);
      % else
        % targetout = 0.5*exp(w3probs*w_class);
      % end
      % targetout = targetout./repmat(sum(targetout,2),1,numtargets);

      % [I J]=max(targetout,[],2);
      % [I1 J1]=max(target,[],2);
      % counter=counter+length(find(J==J1));
      % err_cr = err_cr- sum(sum( target(:,1:end).*log(targetout))) ;
    % end

     % save mnistclassify_weights w1 w2 w3 w_class
     % save mnistclassify_error test_err test_crerr train_err train_crerr mintesterr;

    % end

     % test_err(epoch)=(testnumcases*testnumbatches-counter);
     % test_crerr(epoch)=err_cr/testnumbatches;
     % fprintf(LM_strLearningPrcocessPrvt.hFidLog,'Before epoch %d Train # misclassified: %d (from %d). Test # misclassified: %d (from %d) \t \t \n',...
                % epoch,train_err(epoch),numcases*numbatches,test_err(epoch),testnumcases*testnumbatches);
     % fprintf(1,'Before epoch %d Train # misclassified: %d (from %d). Test # misclassified: %d (from %d) \t \t \n',...
                % epoch,train_err(epoch),numcases*numbatches,test_err(epoch),testnumcases*testnumbatches);
     % if test_err(epoch) < mintesterr
         % w1_min = w1;
         % w2_min = w2;
         % w3_min = w3;
         % w_class_min = w_class;
         if mapping == 1 & keep_min == 1
            for layer = 1 : NM_strNetParams.nNumLayers
                NW_weights{layer} = BP_minWeights{layer};
            end
            w_class = w_class_min;
         end
     % end
    end
    for layer = 1 : NM_strNetParams.nNumLayers
        NW_weights{layer} = BP_minWeights{layer};
    end
    w_class = w_class_min;

     save mnistclassify_weights NW_weights w_class
     save mnistclassify_error test_err test_crerr train_err train_crerr mintesterr;

end

