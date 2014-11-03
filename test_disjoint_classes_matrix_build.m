clear;
clc;
load input_data mTrainTargets mTestTargets;
mTargets = [mTrainTargets; mTestTargets];
nNumTargets = size(mTrainTargets, 2);
mDisjointClassTargets = zeros(nNumTargets, nNumTargets);
xlsFileName = 'class_groups.xlsx';
% Build zero occurence classes to avoid
vZeroOccured = zeros(nNumTargets, 1);
% Build Joint matrix
for i = 1 : size(mTargets, 1)
    vJointClassesIndices = find(mTargets(i,:)==1);
    
    % Update joint recurrence of those classes in the same target
    for n = 1 : size(vJointClassesIndices, 2)
        vZeroOccured(vJointClassesIndices(n)) = vZeroOccured(vJointClassesIndices(n)) + 1;
        for m = 1 : size(vJointClassesIndices, 2) % it could be m = n : size(vJointClassesIndices, 1), but we intend to make it square matrix          
            mDisjointClassTargets (vJointClassesIndices(n), vJointClassesIndices(m)) = mDisjointClassTargets (vJointClassesIndices(n), vJointClassesIndices(m)) + 1;
        end
    end
    
end



% Build POS Names
POSNames = {
'N/A';
'NullPrefix';
'Interrog';
'Conj';
'Confirm';
'Prepos';
'Interj';
'Definit';
'Future';
'ParticleNAASSIB';
'Present';
'Imperative';
'Active';
'Passive';
'Noun';
'NounInfinit';
'NounInfinitLike';
'SubjNoun';
'ExaggAdj';
'ObjNoun';
'TimeLocNoun';
'NoSARF';
'PrepPronComp';
'RelPro';
'DemoPro';
'InterrogArticle';
'JAAZIMA';
'CondJAAZIMA';
'CondNotJAAZIMA';
'LAA';
'N/A';
'Except';
'NoSyntaEffect';
'DZARF';
'ParticleNAASIKH';
'VerbNAASIKH';
'MASSDARIYYA';
'Verb';
'Intransitive';
'Past';
'PresImperat';
'MAJZ';
'Plural';
'MARF';
'MANSS';
'MANS_MAJZ';
'NullSuffix';
'RelAdj';
'Femin';
'PossessPro';
'Masc';
'Single';
'Binary';
'Adjunct';
'NonAdjunct';
'MANSS_MAGR';
'MAGR';
'ObjPossPro';
'SubjPro';
'ObjPro';
'N/A';
'NOUN';
};

% Build the disjoint groups
for i = 1 : size(mDisjointClassTargets, 1)
    mDisjointGroup{i} = find(mDisjointClassTargets(i,i:end) == 0);
    
    % Map to names
    for j = 1 : size(mDisjointGroup{i}, 2)
       mDisjointGroupNames{i}(j) =  POSNames(mDisjointGroup{i}(j));
    end
    
end

% Write to Excel
for i = 1 : size(mDisjointGroupNames, 1)
    xlswrite(xlsFileName, mDisjointGroup{i}, 'GroupNumbers', ['A' num2str(i) ': BC' num2str(i)]);
    xlswrite(xlsFileName, mDisjointGroupNames{i}, 'GroupNames', ['A' num2str(i) ': BC' num2str(i)]);
end

save disjoint_matrix.mat;