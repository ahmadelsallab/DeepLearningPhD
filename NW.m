if ~(mapping == 1 & CONFIG_depthBaseUnitMapping == 1)
	[numcases numdims numbatches]=size(batchdata);
	numhid = NM_strNetParams.vLayersWidths(layer);
	 if restart == 1
	     restart=0;
	     epoch=1;
	     % Initializing symmetric weights and biases. 
	 	if mapping == 1
			if CONFIG_strParams.bBreadthMapping == 1
				if nn_classifier == 1 | svm_classifier == 1
					if (layer == 1)
						vishid = [NW_weights{layer}(1:end-1,:) NW_weights{layer}(1:end-1,:)];
					else
						x = NW_weights{layer}(1:end-1,:);
						vishid = [NW_weights{layer}(1:end-1,:) zeros(size(x,1), size(x,2)); zeros(size(x,1), size(x,2)) NW_weights{layer}(1:end-1,:)];
					end	
					hidbiases  = [NW_weights{layer}(end,:) NW_weights{layer}(end,:)];
					visbiases  = zeros(1,numdims);
				elseif semi_random_mapping == 1
					x = NW_weights{layer}(1:end-1,:);
					vishid     = [NW_weights{layer}(1:end-1,:) 0.1*randn(size(x,1), size(x,2))];
					hidbiases  = [NW_weights{layer}(end,:) NW_weights{layer}(end,:)];
					visbiases  = zeros(1,numdims);
				elseif full_random_mapping == 1
					x = NW_weights{layer}(1:end-1,:);
					vishid     = [0.1*randn(size(x,1), size(x,2)) 0.1*randn(size(x,1), size(x,2))];
					hidbiases  = [0.1*randn(1,size(x,2)) 0.1*randn(1,size(x,2))];
					visbiases  = zeros(1,numdims);
				elseif svm_classifier == 1
					x = NW_weights{layer}(1:end-1,:);
					if (layer == 1)
					  vishid = [NW_weights{layer}(1:end-1,:) NW_weights{layer}(1:end-1,:)];
					else
						vishid = [NW_weights{layer}(1:end-1,:) zeros(size(x,1), size(x,2)); zeros(size(x,1), size(x,2)) NW_weights{layer}(1:end-1,:)];
					end
					hidbiases  = [NW_weights{layer}(end,:) NW_weights{layer}(end,:)];
					visbiases  = zeros(1, numdims);
				elseif weak_classifier == 1
					x = NW_weights{layer}(1:end-1,:);
					if (layer == 1)
					  vishid = [NW_weights{layer}(1:end-1,:) zeros(size(x,1), size(x,2))];
					else
						vishid = [NW_weights{layer}(1:end-1,:) zeros(size(x,1), size(x,2)); zeros(size(x,1), size(x,2)) zeros(size(x,1), size(x,2))];
					end
					hidbiases  = [NW_weights{layer}(end,:) zeros(1,size(x,2))];
					visbiases  = zeros(1,numdims);
				end
			elseif CONFIG_strParams.bDepthMapping == 1
				if CONFIG_NN_depthClassifier == 1
					if  (layer == NM_strNetParams.nPrevNumLayers + 1)
						vishid     = 0.1*randn(numdims, numhid);
						hidbiases  = zeros(1, numhid);
						visbiases  = zeros(1, numdims);
					else
						vishid = NW_weights{mod(layer-1, NM_strNetParams.nPrevNumLayers) + 1}(1:end-1,:);
						hidbiases = NW_weights{mod(layer-1, NM_strNetParams.nPrevNumLayers) + 1}(end,:);
						visbiases  = zeros(1,numdims);
					end
				elseif CONFIG_depthCascadedDataRepresentation == 1
					% Normal random initialization just as if no mapping
	                if(layer == 1)
	                    numdims = size(NW_weights{NM_strNetParams.nPrevNumLayers}, 2);
	                    vishid     = 0.1*randn(numdims, numhid);                    
	                    hidbiases  = zeros(1, numhid);
	                else
							if CONFIG_depthCascadedDataRepReplicated == 1
	                    vishid = NW_weights{layer}(1:end-1,:);
	                    hidbiases = NW_weights{layer}(end,:);
							elseif CONFIG_depthCascadedDataRepRandomize == 1
	                    vishid     = 0.1*randn(numdims, numhid);                    
	                    hidbiases  = zeros(1, numhid);
	                  end
	                end
	            end
	            
	            visbiases  = zeros(1, numdims);
            end
			
        else        
			vishid     = 0.1*randn(numdims, numhid);
			hidbiases  = zeros(1, numhid);
			visbiases  = zeros(1, numdims);
	 	end
		poshidprobs = zeros(numcases, numhid);
		neghidprobs = zeros(numcases, numhid);
		posprods    = zeros(numdims, numhid);
		negprods    = zeros(numdims, numhid);
		vishidinc  = zeros(numdims, numhid);
		hidbiasinc = zeros(1, numhid);
		visbiasinc = zeros(1, numdims);
		batchposhidprobs=zeros(numcases, numhid ,numbatches);
     end
     NW_weights{layer}=[vishid; hidbiases];
else
	NW_unitWeights{unit} = NW_baseUnitWeights;
	
	if unit ~= 1
		numdims = size(NW_baseUnitWeights{end}, 2)+1;
		numhid = size(NW_baseUnitWeights{1}, 2);
		NW_unitWeights{unit}{1} = 0.1*randn(numdims, numhid);
	end
	
	NW_weights{unit} = NW_unitWeights{unit}{1};
end
