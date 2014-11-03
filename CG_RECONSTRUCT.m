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

function [f, df] = CG_RECONSTRUCT(VV,Dim,XX)

l = Dim';
N = size(XX,1);

% Do decomversion.
N_layers = size(Dim, 1) - 1; % remove input layers
offset = 0;
for layer = 1 : N_layers
    NW_weights{layer} = reshape(VV((offset+1) : (offset+(l(layer)+1)*l(layer+1))), l(layer)+1, l(layer+1));
    offset = offset + (l(layer)+1)*l(layer+1);
end

%   XX = [XX ones(N,1)];
%   w1probs = 1./(1 + exp(-XX*w1)); w1probs = [w1probs  ones(N,1)];
%   w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
%   w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
%   w4probs = w3probs*w4; w4probs = [w4probs  ones(N,1)];
%   w5probs = 1./(1 + exp(-w4probs*w5)); w5probs = [w5probs  ones(N,1)];
%   w6probs = 1./(1 + exp(-w5probs*w6)); w6probs = [w6probs  ones(N,1)];
%   w7probs = 1./(1 + exp(-w6probs*w7)); w7probs = [w7probs  ones(N,1)];
%   XXout = 1./(1 + exp(-w7probs*w8));
%   
BP_layerInputData = XX;
XX = [XX ones(N,1)];    

% Feed Fwd                       
[mTempNotAugActivationData mAugActivationData] = NM_neuralNetActivation(BP_layerInputData, NW_weights(1 : N_layers/2 - 1));

%w4probs = w3probs*w4; w4probs = [w4probs  ones(N,1)];
mAugActivationData{N_layers/2} = mAugActivationData{N_layers/2 - 1} * NW_weights{N_layers/2};
%mAugActivationData{N_layers/2} = [mAugActivationData{N_layers/2} ones(N, 1)];Augmentation shall be done inside NM_neuralNetActivation


% Reconstruct
[mTempNotAugActivationData mReconstructedBatchData] = NM_neuralNetActivation(mAugActivationData{N_layers/2}, NW_weights(N_layers/2 + 1 : N_layers));
mAugActivationData{N_layers/2} = [mAugActivationData{N_layers/2} ones(N, 1)];

% Compute the reconstruction error
% f = -1/N*sum(sum( XX(:,1:end-1).*log(XXout) + (1-XX(:,1:end-1)).*log(1-XXout)));
% IO = 1/N*(XXout-XX(:,1:end-1));
% Ix8=IO; 
% dw8 =  w7probs'*Ix8;
% 
% Ix7 = (Ix8*w8').*w7probs.*(1-w7probs); 
% Ix7 = Ix7(:,1:end-1);
% dw7 =  w6probs'*Ix7;
% 
% Ix6 = (Ix7*w7').*w6probs.*(1-w6probs); 
% Ix6 = Ix6(:,1:end-1);
% dw6 =  w5probs'*Ix6;
% 
% Ix5 = (Ix6*w6').*w5probs.*(1-w5probs); 
% Ix5 = Ix5(:,1:end-1);
% dw5 =  w4probs'*Ix5;
% 
% Ix4 = (Ix5*w5');
% Ix4 = Ix4(:,1:end-1);
% dw4 =  w3probs'*Ix4;
% 
% Ix3 = (Ix4*w4').*w3probs.*(1-w3probs); 
% Ix3 = Ix3(:,1:end-1);
% dw3 =  w2probs'*Ix3;
% 
% Ix2 = (Ix3*w3').*w2probs.*(1-w2probs); 
% Ix2 = Ix2(:,1:end-1);
% dw2 =  w1probs'*Ix2;
% 
% Ix1 = (Ix2*w2').*w1probs.*(1-w1probs); 
% Ix1 = Ix1(:,1:end-1);
% dw1 =  XX'*Ix1;

f = -1 / N * (sum(sum( (BP_layerInputData - mReconstructedBatchData{N_layers/2}(:, 1:end-1)).^2 )));

IO = 1/N*(mReconstructedBatchData{N_layers/2}(:,1:end-1) - BP_layerInputData);

BP_wprobs = [mAugActivationData mReconstructedBatchData];


layer = N_layers - 1;
Ix_upper = IO;
Ix{N_layers} = IO;
dw{N_layers} = BP_wprobs{layer}'*Ix{N_layers};

w_upper = NW_weights{N_layers};
while (layer >= 1)
    Ix{layer} = (Ix_upper*w_upper').*BP_wprobs{layer}.*(1-BP_wprobs{layer});
    Ix{layer} = Ix{layer}(:,1:end-1);
    if(layer ~= 1)
        dw{layer} = (BP_wprobs{layer-1})'*Ix{layer};
    else
        dw{layer} = XX'*Ix{layer};
    end
    Ix_upper = [];
    Ix_upper = Ix{layer};
    w_upper = [];
    w_upper = NW_weights{layer};
    layer = layer - 1;
end

% df = [dw1(:)' dw2(:)' dw3(:)' dw4(:)' dw5(:)' dw6(:)'  dw7(:)'  dw8(:)'  ]'; 
df = [];
for(layer = 1 : N_layers)
	df = [df dw{layer}(:)'];
end
df = df';



