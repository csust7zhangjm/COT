function [ output_args ] = convrt2PlotOutput( )
%CONVRT_OUTPUT Summary of this function goes here
%   Detailed explanation goes here

bbreg = false;
seqList = importdata('../dataset_otb/seq_list_old.txt');
methodList = {'GCNN', 'GCNN_avg_sameweight', 'GCN_maxonly'};

for i=1:length(seqList)
    for j=1:length(methodList)
        
        seqName = seqList{i};
        config = genConfig('otb', seqName, methodList{j});
        
        switch(seqName)
            case{'FaceOcc1'}
                seqName = 'Faceocc1';
            case{'FaceOcc2'}
                seqName = 'Faceocc2';
            case{'FleetFace'}
                seqName = 'Fleetface';
        end
        
        if bbreg
            resultDir = fullfile(pwd,'plotResult',[methodList{j} '_bb']);
            resultName = [lower(seqName(1)) seqName(2:end) '_' methodList{j} '_bb.mat'];
            bboxPath = fullfile(config.Dir.result, 'result_0.mat');
        else
            resultDir = fullfile(pwd,'plotResult',methodList{j});
            resultName = [lower(seqName(1)) seqName(2:end) '_' methodList{j} '.mat'];
            bboxPath = fullfile(config.Dir.result, 'result.mat');
        end
        if ~exist(resultDir, 'dir'), mkdir(resultDir); end
        
        if(~exist(bboxPath,'file'))
            fprintf('%s does not exist\n',bboxPath);
            continue;
        end
        
        len = length(config.Dir.imgs);
        
        res = load(bboxPath);
        
        results = cell(1);
        results{1}.res = res.result;
        results{1}.type = 'rect';
        results{1}.len = len;
        switch(config.Name.seq)
            case{'David'}
                results{1}.annoBegin = 300;
                results{1}.startFrame = 300;
            case{'Tiger1'}
                results{1}.annoBegin = 1;
                results{1}.startFrame = 6;
            otherwise
                results{1}.annoBegin = 1;
                results{1}.startFrame = 1;
        end
        save(fullfile(resultDir,resultName), 'results');
    end
end


end

