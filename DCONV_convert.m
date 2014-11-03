% Function:
% Converts the input txt file into features and targets MATLAB vectors.
% Inputs:
% sFileName: The name of the txt file, organized such that the odd lines are targets and even ones are features.
% eFeaturesMode: Raw, Normal, Binary, Bitfield
% Output:
% mFeatures: Matrix (nxm), where n is the number of examples and m is the features vector length
% mTargets: Matrix (nxl), where n is the number of examples and l is the number of target classes
% nBitfieldLength: The bitfield length in case of Raw
% vChunkLength: Vector storing the boundaries of each features chunk
% vOffset: The bitfield boundaries corresponding to each chunk
function [mFeatures, mTargets, nBitfieldLength, vChunkLength, vOffset] = DCONV_convert(sFileName, eFeaturesMode)
    
    % Open the file
    hFid = fopen(sFileName);

    % Initialize the matrices
    mFeatures = [];
    mTargets = [];

    % This flag indicates if the next line is target (=0) or feature (=1)
    bTarget = 1;

    % Read the first line
    sLine = fgets(hFid);
    
    
    switch(eFeaturesMode)
        case 'Raw'
            % The first line is the bitfield length
            nBitfieldLength = str2num(sLine);
            sLine = fgets(hFid);
            
            % The sencond line is the features chunck length
            vChunkLength = str2num(sLine);
            sLine = fgets(hFid);

            % The third line is the offsets bitfield boundaries
            vOffset = str2num(sLine);
            sLine = fgets(hFid);
            
        otherwise
            nBitfieldLength = 0;
    end % end switch   
    
    nLineCtr = 0;
    nExampleCtr = 0;

    % Loop on all lines
    while(sLine > 0)
        
        nLineCtr = nLineCtr + 1;
        if bTarget == 0
            % Feature line--> Concatenate features matrix
            mFeatures = [mFeatures; str2num(sLine)];
            nExampleCtr = nExampleCtr + 1;
            fprintf(1, 'Conversion of example %d is done successfully\n', nExampleCtr);
        else
            % Target line--> Concatenate targets matrix
            mTargets = [mTargets; str2num(sLine)];
        end

        % Toggle target flag
        bTarget = ~bTarget;

        % Read next line
        sLine = fgets(hFid);
    end;

    % Close the file
    fclose(hFid);

end % end function