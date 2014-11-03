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

% This program reads raw MNIST files available at 
% http://yann.lecun.com/exdb/mnist/ 
% and converts them to files in matlab format 
% Before using this program you first need to download files:
% train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz 
% t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz
% and gunzip them. You need to allocate some space for this.  

% This program was originally written by Yee Whye Teh

% Function: Modified from Ruslan Salakhutdinov and Geoff Hinton (See copyright above) 
% Converts the input NIST files into train and test features and mTrainmTestTargets vector
% Inputs:
% CONFIG_strParams: Reference to the configurations parameters structure
% Output:
% mTrainFeatures, mTrainmTrainmTestTargets, mTestFeatures, mTestmTrainmTestTargets save into CONFIG_strParams.sInputDataWorkspace
function DCONV_convertMNIST(CONFIG_strParams)

	% Read raw files and save each number class in .mat file
	f = fopen([CONFIG_strParams.sDatasetFilesPath '\t10k-images-idx3-ubyte'],'r');
	[a,count] = fread(f,4,'int32');
	  
	g = fopen([CONFIG_strParams.sDatasetFilesPath '\t10k-labels-idx1-ubyte'],'r');
	[l,count] = fread(g,2,'int32');

	fprintf(1,'Starting to convert Test MNIST images (prints 10 dots) \n'); 
	n = 1000;

	Df = cell(1,10);
	for d=0:9,
	  Df{d+1} = fopen(['test' num2str(d) '.ascii'],'w');
	end;
	  
	for i=1:10,
	  fprintf('.');
	  rawimages = fread(f,28*28*n,'uchar');
	  rawlabels = fread(g,n,'uchar');
	  rawimages = reshape(rawimages,28*28,n);

	  for j=1:n,
		fprintf(Df{rawlabels(j)+1},'%3d ',rawimages(:,j));
		fprintf(Df{rawlabels(j)+1},'\n');
	  end;
	end;

	fprintf(1,'\n');
	for d=0:9,
	  fclose(Df{d+1});
	  D = load(['test' num2str(d) '.ascii'],'-ascii');
	  fprintf('%5d Digits of class %d\n',size(D,1),d);
	  save(['test' num2str(d) '.mat'],'D','-mat');
	end;


	% Work with trainig files second  
	f = fopen([CONFIG_strParams.sDatasetFilesPath '\train-images-idx3-ubyte'],'r');
	[a,count] = fread(f,4,'int32');

	g = fopen([CONFIG_strParams.sDatasetFilesPath '\train-labels-idx1-ubyte'],'r');
	[l,count] = fread(g,2,'int32');

	fprintf(1,'Starting to convert Training MNIST images (prints 60 dots)\n'); 
	n = 1000;

	Df = cell(1,10);
	for d=0:9,
	  Df{d+1} = fopen(['digit' num2str(d) '.ascii'],'w');
	end;

	for i=1:60,
	  fprintf('.');
	  rawimages = fread(f,28*28*n,'uchar');
	  rawlabels = fread(g,n,'uchar');
	  rawimages = reshape(rawimages,28*28,n);

	  for j=1:n,
		fprintf(Df{rawlabels(j)+1},'%3d ',rawimages(:,j));
		fprintf(Df{rawlabels(j)+1},'\n');
	  end;
	end;

	fprintf(1,'\n');
	for d=0:9,
	  fclose(Df{d+1});
	  D = load(['digit' num2str(d) '.ascii'],'-ascii');
	  fprintf('%5d Digits of class %d\n',size(D,1),d);
	  save(['digit' num2str(d) '.mat'],'D','-mat');
	end;

	dos('rm *.ascii');
	
    clear Df D rawimages rawlabels f a count l i j n d;
	
	% Convert train data and mTrainmTestTargets
	mTrainFeatures=[]; 
	mTrainTargets=[]; 
	
	
	for i = 1 : size(CONFIG_strParams.vSubClassTargets, 2) 
		load(['digit' num2str(CONFIG_strParams.vSubClassTargets(i)) '.mat']);
		mTrainFeatures = [mTrainFeatures; D];
		vTempTargets = zeros(1, length(CONFIG_strParams.vSubClassTargets));
		vTempTargets(i) = 1;
		mTrainTargets = [mTrainTargets; repmat(vTempTargets, size(D,1), 1)];
	end
	% load digit0; mTrainFeatures = [mTrainFeatures; D]; mTrainmTestTargets = [mTrainmTestTargets; repmat([1 0 0 0 0 0 0 0 0 0], size(D,1), 1)];  
	% load digit1; mTrainFeatures = [mTrainFeatures; D]; mTrainmTestTargets = [mTrainmTestTargets; repmat([0 1 0 0 0 0 0 0 0 0], size(D,1), 1)];
	% load digit2; mTrainFeatures = [mTrainFeatures; D]; mTrainmTestTargets = [mTrainmTestTargets; repmat([0 0 1 0 0 0 0 0 0 0], size(D,1), 1)]; 
	% load digit3; mTrainFeatures = [mTrainFeatures; D]; mTrainmTestTargets = [mTrainmTestTargets; repmat([0 0 0 1 0 0 0 0 0 0], size(D,1), 1)];
	% load digit4; mTrainFeatures = [mTrainFeatures; D]; mTrainmTestTargets = [mTrainmTestTargets; repmat([0 0 0 0 1 0 0 0 0 0], size(D,1), 1)]; 
	% load digit5; mTrainFeatures = [mTrainFeatures; D]; mTrainmTestTargets = [mTrainmTestTargets; repmat([0 0 0 0 0 1 0 0 0 0], size(D,1), 1)];
	% load digit6; mTrainFeatures = [mTrainFeatures; D]; mTrainmTestTargets = [mTrainmTestTargets; repmat([0 0 0 0 0 0 1 0 0 0], size(D,1), 1)];
	% load digit7; mTrainFeatures = [mTrainFeatures; D]; mTrainmTestTargets = [mTrainmTestTargets; repmat([0 0 0 0 0 0 0 1 0 0], size(D,1), 1)];
	% load digit8; mTrainFeatures = [mTrainFeatures; D]; mTrainmTestTargets = [mTrainmTestTargets; repmat([0 0 0 0 0 0 0 0 1 0], size(D,1), 1)];
	% load digit9; mTrainFeatures = [mTrainFeatures; D]; mTrainmTestTargets = [mTrainmTestTargets; repmat([0 0 0 0 0 0 0 0 0 1], size(D,1), 1)];
	mTrainFeatures = mTrainFeatures/255;
	
	mTestFeatures=[];
	mTestTargets=[];
	for i = 1 : size(CONFIG_strParams.vSubClassTargets, 2) 
		load(['test' num2str(CONFIG_strParams.vSubClassTargets(i)) '.mat']);
		mTestFeatures = [mTestFeatures; D];
		vTempTargets = zeros(1, length(CONFIG_strParams.vSubClassTargets));
		vTempTargets(i) = 1;
		mTestTargets = [mTestTargets; repmat(vTempTargets, size(D,1), 1)];
	end
	% load test0; mTestFeatures = [mTestFeatures; D]; mTestTargets = [mTestTargets; repmat([1 0 0 0 0 0 0 0 0 0], size(D,1), 1)]; 
	% load test1; mTestFeatures = [mTestFeatures; D]; mTestTargets = [mTestTargets; repmat([0 1 0 0 0 0 0 0 0 0], size(D,1), 1)]; 
	% load test2; mTestFeatures = [mTestFeatures; D]; mTestTargets = [mTestTargets; repmat([0 0 1 0 0 0 0 0 0 0], size(D,1), 1)];
	% load test3; mTestFeatures = [mTestFeatures; D]; mTestTargets = [mTestTargets; repmat([0 0 0 1 0 0 0 0 0 0], size(D,1), 1)];
	% load test4; mTestFeatures = [mTestFeatures; D]; mTestTargets = [mTestTargets; repmat([0 0 0 0 1 0 0 0 0 0], size(D,1), 1)];
	% load test5; mTestFeatures = [mTestFeatures; D]; mTestTargets = [mTestTargets; repmat([0 0 0 0 0 1 0 0 0 0], size(D,1), 1)];
	% load test6; mTestFeatures = [mTestFeatures; D]; mTestTargets = [mTestTargets; repmat([0 0 0 0 0 0 1 0 0 0], size(D,1), 1)];
	% load test7; mTestFeatures = [mTestFeatures; D]; mTestTargets = [mTestTargets; repmat([0 0 0 0 0 0 0 1 0 0], size(D,1), 1)];
	% load test8; mTestFeatures = [mTestFeatures; D]; mTestTargets = [mTestTargets; repmat([0 0 0 0 0 0 0 0 1 0], size(D,1), 1)];
	% load test9; mTestFeatures = [mTestFeatures; D]; mTestTargets = [mTestTargets; repmat([0 0 0 0 0 0 0 0 0 1], size(D,1), 1)];
	mTestFeatures = mTestFeatures/255;
    
    save(CONFIG_strParams.sInputDataWorkspace, '-v7.3', 'mTestFeatures', 'mTestTargets', 'mTrainFeatures', 'mTrainTargets');
end % end function
