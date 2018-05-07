function [ config ] = genConfig(dataset,seqName,methodName)
%
% Generate a configuration of an OTB sequence
%
% Hyeonseob Nam, 2015
% namhs09@postech.ac.kr

% directories
config.Dir.Home = pwd;

% Names
config.Name.seq = seqName;
if exist('methodName','var'), config.Name.method = methodName; end

switch(dataset)
    case 'otb'
        benchmarkSeqHome ='./dataset_otb';
        resultHome = './result';
        
        % img path
        switch(config.Name.seq)
            case {'Jogging-1', 'Jogging-2'}
                config.Dir.seq = fullfile(benchmarkSeqHome, 'Jogging', 'img');
            case {'Skating2-1', 'Skating2-2'}
                config.Dir.seq = fullfile(benchmarkSeqHome, 'Skating2', 'img');
            otherwise
                config.Dir.seq = fullfile(benchmarkSeqHome, config.Name.seq, 'img');
        end
        
        if(~exist(config.Dir.seq,'file'))
            error('%s does not exist!!',config.Dir.seq);
        end
        
        % parse img list
        config.Dir.imgs = parseImg(config.Dir.seq);
        switch(config.Name.seq)
            case 'David'
                config.Dir.imgs = config.Dir.imgs(300:end);
            case 'Tiger1'
                config.Dir.imgs = config.Dir.imgs(6:end);
        end
        
        % load gt
        switch(config.Name.seq)
            case 'Jogging-1'
                gtPath = fullfile(benchmarkSeqHome, 'Jogging', 'groundtruth_rect.1.txt');
            case 'Jogging-2'
                gtPath = fullfile(benchmarkSeqHome, 'Jogging', 'groundtruth_rect.2.txt');
            case 'Skating2-1'
                gtPath = fullfile(benchmarkSeqHome, 'Skating2', 'groundtruth_rect.1.txt');
            case 'Skating2-2'
                gtPath = fullfile(benchmarkSeqHome, 'Skating2', 'groundtruth_rect.2.txt');
            case 'Human4'
                gtPath = fullfile(benchmarkSeqHome, 'Human4', 'groundtruth_rect.2.txt');
            otherwise
                gtPath = fullfile(benchmarkSeqHome, config.Name.seq, 'groundtruth_rect.txt');
        end
        
        if(exist(gtPath,'file'))
            gt = importdata(gtPath);
            switch(config.Name.seq)
                case 'Tiger1'
                    gt = gt(6:end,:);
                case {'Board','Twinnings'}
                    gt = gt(1:end-1,:);
            end
            config.Data.gt = gt;
        else
            error('%s does not exist!!',gtPath);
        end
        
        nFrames = min(length(config.Dir.imgs), size(config.Data.gt,1));
        config.Dir.imgs = config.Dir.imgs(1:nFrames);
        config.Data.gt = config.Data.gt(1:nFrames,:);
        
        if exist('methodName','var')
            config.Dir.result = fullfile(resultHome, methodName, config.Name.seq);
%             config.DirBBreg.result = fullfile(resultHome, [methodName '_bbP1'], config.Name.seq);
%             config.DirMixed.result = fullfile(resultHome, [methodName '_mixed'], config.Name.seq);
            genDir(config.Dir.result);
%             genDir(config.DirBBreg.result);
%             genDir(config.DirMixed.result);
        end
        
    case {'vot2013','vot2014','vot2015'}
        benchmarkSeqHome = ['../GCNN/dataset_vot/', dataset(end-3:end)];
        
        % img path
        config.Dir.seq = fullfile(benchmarkSeqHome, config.Name.seq);
        if(~exist(config.Dir.seq,'file'))
            error('%s does not exist!!',config.Dir.seq);
        end
        
        % parse img list
        images = dir(fullfile(config.Dir.seq,'*.jpg'));
        images = {images.name}';
        images = cellfun(@(x) fullfile(config.Dir.seq,x), images, 'UniformOutput', false);
        config.Dir.imgs = images;
        
        % gt path
        gtPath = fullfile(benchmarkSeqHome, config.Name.seq, 'groundtruth.txt');
        if(~exist(gtPath,'file'))
            error('%s does not exist!!',gtPath);
        end
        
        % parse gt
        gt = importdata(gtPath);
        if size(gt,2) >= 6
            x = gt(:,1:2:end);
            y = gt(:,2:2:end);
            gt = [min(x,[],2), min(y,[],2), max(x,[],2) - min(x,[],2), max(y,[],2) - min(y,[],2)];
        end
        config.Data.gt = gt;
        
        nFrames = min(length(config.Dir.imgs), size(config.Data.gt,1));
        config.Dir.imgs = config.Dir.imgs(1:nFrames);
        config.Data.gt = config.Data.gt(1:nFrames,:);
        
        resultHome = './result_vot';
        if exist('methodName','var')
            config.Dir.result = fullfile(resultHome, methodName, config.Name.seq);
            genDir(config.Dir.result);
        end        
end

function genDir(path)
if ~exist(path,'dir')
    mkdir(path);
end
