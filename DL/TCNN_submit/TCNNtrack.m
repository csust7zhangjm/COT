function TCNNtrack
% ncc VOT integration example
% 
% This function is an example of tracker integration into the toolkit.
% The implemented tracker is a very simple NCC tracker that is also used as
% the baseline tracker for challenge entries.
%

% *************************************************************
% VOT: Always call exit command at the end to terminate Matlab!
% *************************************************************
cleanup = onCleanup(@() exit() );

% *************************************************************
% VOT: Set random seed to a different value every time.
% *************************************************************
RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', sum(clock)));

% **********************************
% VOT: Get initialization data
% **********************************
[handle, image, region] = vot('rectangle');

%% SETUP
run setup_gamma1.m;

% Initialize the tracker
[state, ~] = TCNN_initialize(imread(image), region);

while true

    % **********************************
    % VOT: Get next frame
    % **********************************
    [handle, image] = handle.frame(handle);

    if isempty(image)
        break;
    end;
    
	% Perform a tracking step, obtain new region
    [state, region] = TCNN_update(state, imread(image));
    
    % **********************************
    % VOT: Report position for frame
    % **********************************
    handle = handle.report(handle, region);
    
end;

% **********************************
% VOT: Output the results
% **********************************
handle.quit(handle);

end

function [state, location] = TCNN_initialize(I, region, varargin)

    %% GCNN_INIT
    % use gpu
    state.opts.useGpu = true;

    % bounding box regression
    state.opts.bbreg = true;
    state.opts.bbreg_nSamples = 1000;

    % model def
    state.opts.net_file = fullfile('/home/mooyeol/Documents/MATLAB/VOTnew16/TCNN_submit/TCNN/models/imagenet-vgg-m_conv3_512-512-2.mat');

    % learning policy
    state.opts.batchSize = 128;
    state.opts.batch_pos = 32;
    state.opts.batch_neg = 96;

    % test policy
    state.opts.batchSize_test = 256;

    % initial training policy
    state.opts.learningRate_init = 0.001;
    state.opts.maxiter_init = 50;

    state.opts.nPos_init = 500;
    state.opts.nNeg_init = 5000;
    state.opts.posThr_init = 0.7;
    state.opts.negThr_init = 0.5;

    % update policy
    state.opts.learningRate_update = 0.003;
    state.opts.maxiter_update = 10;

    state.opts.nPos_update = 50;
    state.opts.nNeg_update = 200;
    state.opts.posThr_update = 0.7;
    state.opts.negThr_update = 0.3;

    % block policy
    state.opts.block_frames = 10;
    state.opts.parent_blocks = 10;

    % cropping policy
    state.opts.input_size = 75;
    state.opts.crop_mode = 'wrap';
    state.opts.crop_padding = 8;

    % scaling policy
    state.opts.scale_factor = 1.05;

    % sampling policy
    state.opts.nSamples = 256;
    state.opts.trans_f = 0.6; % translation std: mean(width,height)*trans_f/2
    state.opts.scale_f = 1; % scaling std: scale_factor^(scale_f/2)

    % set image size
    img = I;
    state.opts.imgSize = size(img);

    %% load net
    net = load(state.opts.net_file);
    if isfield(net,'net'), net = net.net; end
    state.net_conv.layers = net.layers(1:10);
    state.net_fc.layers = net.layers(11:end);
    clear net;

    for i=1:numel(state.net_fc.layers)
        switch (state.net_fc.layers{i}.name)
            case {'fc3','fc4','fc5'}
                state.net_fc.layers{i}.filtersLearningRate = 1;
                state.net_fc.layers{i}.biasesLearningRate = 2;
            case {'fc6'}
                state.net_fc.layers{i}.filtersLearningRate = 1;
                state.net_fc.layers{i}.biasesLearningRate = 2;
        end
    end
    
    if state.opts.useGpu
        state.net_conv = vl_simplenn_move(state.net_conv, 'gpu') ;
        state.net_fc = vl_simplenn_move(state.net_fc, 'gpu') ;
    else
        state.net_conv = vl_simplenn_move(state.net_conv, 'cpu') ;
        state.net_fc = vl_simplenn_move(state.net_fc, 'cpu') ;
    end
    
    %% GCNN_INTERFACE
    state.chooseWeight = 'none';
    state.chooseSample = 'all';
    state.choose = 'weighted';
    state.chooseMax = 'stochastic';
    state.chooseScoring = 'weightedSum';
    
    if(size(img,3)==1), img = cat(3,img,img,img); end

    % If the provided region is a polygon ...
    if numel(region) > 4
        x1 = round(min(region(1:2:end)));
        x2 = round(max(region(1:2:end)));
        y1 = round(min(region(2:2:end)));
        y2 = round(max(region(2:2:end)));
        state.targetLoc = round([x1, y1, x2 - x1, y2 - y1]);
    else
        state.targetLoc = round([round(region(1)), round(region(2)), ... 
            round(region(1) + region(3)) - round(region(1)), ...
            round(region(2) + region(4)) - round(region(2))]);
    end;    

    %% training examples
    pos_examples = genSamples('permute', state.targetLoc, state.opts.nPos_init*10, state.opts, 'uniform', 0.1, 5);
    r = overlap_ratio(pos_examples,state.targetLoc);
    pos_examples = pos_examples(r>state.opts.posThr_init,:);
    pos_examples = pos_examples(randsample(length(pos_examples),state.opts.nPos_init),:);

    neg_examples = [genSamples('permute', state.targetLoc, state.opts.nNeg_init, state.opts, 'uniform', 1, 10);...
        genSamples('slide', state.targetLoc, state.opts.nNeg_init, state.opts, 5)];
    r = overlap_ratio(neg_examples,state.targetLoc);
    neg_examples = neg_examples(r<state.opts.negThr_init,:);
    neg_examples = neg_examples(randsample(length(neg_examples),state.opts.nNeg_init),:);

    examples = [pos_examples; neg_examples];
    pos_idx = 1:size(pos_examples,1);
    neg_idx = (1:size(neg_examples,1)) + size(pos_examples,1);

    % extract conv5 features
    feat_conv = sample_features_convX(state.net_conv, img, examples, state.opts);
    pos_data = feat_conv(:,:,:,pos_idx);
    neg_data = feat_conv(:,:,:,neg_idx);

    %% Learning
    [state.net_fc, state.hardnegs] = gnet_finetune_hnm(state.net_fc,pos_data,neg_data,state.opts,...
        'maxiter',state.opts.maxiter_init,'learningRate',state.opts.learningRate_init);
    % [state.net_fc] = gnet_finetune(state.net_fc,pos_data,neg_data,opts,...
    %     'maxiter',state.opts.maxiter_init,'learningRate',state.opts.learningRate_init);

    %% Prepare training data
    state.total_pos_data = cell(1,1,1,1);
    state.total_neg_data = cell(1,1,1,1);

    pos_examples = genSamples('permute', state.targetLoc, state.opts.nPos_update, state.opts, 'gaussian', 0.1, 5);
    r = overlap_ratio(pos_examples,state.targetLoc);
    pos_examples = pos_examples(r>state.opts.posThr_update,:);

    neg_examples = genSamples('permute', state.targetLoc, state.opts.nNeg_update, state.opts, 'uniform', 2, 5);
    r = overlap_ratio(neg_examples,state.targetLoc);
    neg_examples = neg_examples(r<state.opts.negThr_update,:);

    examples = [pos_examples; neg_examples];
    pos_idx = 1:size(pos_examples,1);
    neg_idx = (1:size(neg_examples,1)) + size(pos_examples,1);

    feat_conv = sample_features_convX(state.net_conv, img, examples, state.opts);
    state.total_pos_data{1,1,1,1} = feat_conv(:,:,:,pos_idx);
    state.total_neg_data{1,1,1,1} = feat_conv(:,:,:,neg_idx);

    state.success_frames = 1;
    state.trans_f = state.opts.trans_f;
    state.scale_f = state.opts.scale_f;

    %% init graph structure
    state.blocks = struct('frames',cell(1,1), 'parents',[], 'econfs',[],...
        'rel_parents',[], 'minconfs',[], 'maxminparent',[], 'maxminconf',inf('single'),...
        'cnn',[]);
    state.blocks(1).frames = 1;
    state.blocks(1).cnn = state.net_fc;
    state.cand_blocks = 1;

%     %% initialize displayots
%     if display
%         figure(1);
%         set(gcf,'Position',[200 100 600 800],'MenuBar','none','ToolBar','none');
% 
%         subplot(2,1,1);
%         imshow(img,'initialmagnification','fit'); hold on;
% 
%         rectangle('Position', state.targetLoc, 'EdgeColor', [1 0 0], 'Linewidth', 3);
%         set(gca,'position',[0 0.5 1 0.5]);
% 
%         % frame index
%         text(10,10,'1','Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30);
%         hold off; drawnow; pause(0.001);
% 
%         subplot(2,1,2); plot(1:nFrames,score); grid on; xlim([1,nFrames]); ylim([0,1]);
%         set(gca,'position',[0.1 0.05 0.8 0.4]);
%     end
    state.To = 1;
    state.curBlock = 2;        
    location = state.targetLoc;

    %% Train bbox reg
    if state.opts.bbreg
        state.net_conv_bbreg = state.net_conv;
        state.net_conv_bbreg.layers = state.net_conv_bbreg.layers(1:8);

        pos_examples = genSamples('permute_bboxreg', state.targetLoc, state.opts.bbreg_nSamples*10, state.opts, 0.3, 10);
        r = overlap_ratio(pos_examples,state.targetLoc);
        pos_examples = pos_examples(r>0.6,:);
        pos_examples = pos_examples(randsample(end,min(state.opts.bbreg_nSamples,end)),:);
        feat_conv = sample_features_convX(state.net_conv_bbreg, img, pos_examples, state.opts);

        X = permute(gather(feat_conv),[4,3,1,2]);
        X = X(:,:);
        bbox = pos_examples;
        bbox_gt = repmat(state.targetLoc,size(pos_examples,1),1);
        state.bbox_reg = train_bbox_regressor(X, bbox, bbox_gt);    
    end
end

function [state, location] = TCNN_update(state, I, varargin)
 
    state.To = state.To + 1;
    img = I;
    if(size(img,3)==1), img = cat(3,img,img,img); end
    
    %% estimation
    samples = genSamples('permute', state.targetLoc, state.opts.nSamples, state.opts, 'gaussian', state.trans_f, state.scale_f);
    feat_conv = sample_features_convX(state.net_conv, img, samples, state.opts);
    
    scores = zeros(state.opts.nSamples,length(state.cand_blocks),'single');
    for i=1:length(state.cand_blocks)
        b = state.cand_blocks(i);
        s = sample_features_fcX(state.blocks(b).cnn, feat_conv, state.opts, strcmp(state.chooseWeight, 'none'));
        s = squeeze(s)';
        scores(:,i) = s(:,2);
    end
    econfs = max(scores);
    
    minconfs = min(cell2mat({state.blocks(state.cand_blocks).maxminconf})', econfs');
    
    if strcmp(state.chooseWeight, 'exp')
        weights = exp(10*minconfs);
        weights = weights/sum(weights);
    elseif strcmp(state.chooseWeight, 'none')
        weights = minconfs;
%         weights = weights.*(0.95.^((length(weights)-1):-1:0))';
        weights = weights/sum(weights);
    else
        error('chooseWeight error');
    end
    
    scores_all = scores;

    if strcmp(state.chooseScoring, 'weightedSum')
        scores = scores*weights;
    elseif strcmp(state.chooseScoring, 'max')
        [value, ~] = max(weights, [], 1); 
        scores = max(scores(:,weights==value), [], 2);
%         fprintf('weights: '); fprintf('%.2f ',weights); fprintf('\n');
%         fprintf('max: '); fprintf('%.2f ',value); fprintf('\n');
    elseif strcmp(state.chooseScoring, 'mean')
        scores = mean(scores,2);
    elseif strcmp(state.chooseScoring, 'meanOfMax')
        [value, ~] = max(weights, [], 1); 
        scores = mean(scores(:,weights==value), 2);
%         fprintf('weights: '); fprintf('%.2f ',weights); fprintf('\n');
%         fprintf('max: '); fprintf('%.2f ',value); fprintf('\n');        
    elseif strcmp(state.chooseScoring, 'single')
    else
        error('chooseScoring error')
    end
%         fprintf('parents: '); fprintf('%.2f ',state.cand_blocks); fprintf('\n');
%         fprintf('minconfs: '); fprintf('%.2f ',minconfs); fprintf('\n');
%         fprintf('weights: '); fprintf('%.2f ',weights); fprintf('\n');
    
    [~,idx] = sort(scores,'descend');
    state.targetLoc = round(mean(samples(idx(1:5),:)));
    
    %% BB regression
    if state.opts.bbreg
        feat_conv = sample_features_convX(state.net_conv_bbreg, img, state.targetLoc, state.opts);

        % bbox regression
        X_ = permute(gather(feat_conv(:,:,:,1)),[4,3,1,2]);
        X_ = X_(:,:);
        bbox_ = state.targetLoc;
        pred_boxes = predict_bbox_regressor(state.bbox_reg.model, X_, bbox_);
        location = round(mean(pred_boxes,1));
    else
        location = state.targetLoc;
    end
    
    %% prepare training data
    %     if(score_max>0)
    pos_examples = genSamples('permute', state.targetLoc, state.opts.nPos_update, state.opts, 'gaussian', 0.1, 5);
    r = overlap_ratio(pos_examples,state.targetLoc);
    pos_examples = pos_examples(r>state.opts.posThr_update,:);
    
    neg_examples = genSamples('permute', state.targetLoc, state.opts.nNeg_update, state.opts, 'uniform', 2, 5);
    r = overlap_ratio(neg_examples,state.targetLoc);
    neg_examples = neg_examples(r<state.opts.negThr_update,:);
    
    examples = [pos_examples; neg_examples];
    pos_idx = 1:size(pos_examples,1);
    neg_idx = (1:size(neg_examples,1)) + size(pos_examples,1);
    
    feat_conv = sample_features_convX(state.net_conv, img, examples, state.opts);
    state.total_pos_data{1,1,1,end+1} = feat_conv(:,:,:,pos_idx);
    state.total_neg_data{1,1,1,end+1} = feat_conv(:,:,:,neg_idx);
    
    if (state.To > state.opts.block_frames*(state.opts.parent_blocks+1))
        state.total_pos_data{1,1,1,end-state.opts.block_frames*(state.opts.parent_blocks+1)} = single([]);
    end
    if (state.To > state.opts.block_frames)
        state.total_neg_data{1,1,1,end-state.opts.block_frames} = single([]);
    end

%     feat_conv_ = sample_features_convX(net_conv, img, targetLoc, opts);
%     econf_recalc = zeros(1,length(cand_blocks),'single');
%     for i=1:length(cand_blocks)
%         b = cand_blocks(i);
%         s = sample_features_fcX(blocks(b).cnn, feat_conv_, opts, strcmp(chooseWeight, 'none'));
%         s = squeeze(s)';
%         econf_recalc(:,i) = s(:,2);
%     end
    
    %% update graph
    if size(state.blocks,2) < state.curBlock
        state.blocks(state.curBlock) = struct('frames',cell(1,1), 'parents',[], 'econfs',[],...
        'rel_parents',[], 'minconfs',[], 'maxminparent',[], 'maxminconf',inf('single'),...
        'cnn',[]);
    end        
    state.blocks(state.curBlock).frames = [state.blocks(state.curBlock).frames, state.To];
    econfs_mean = mean(scores_all(idx(1:5),:));
    state.blocks(state.curBlock).econfs = [state.blocks(state.curBlock).econfs; econfs_mean];
    
%     fprintf('\nrecalc: %f   mean: %f\n\n',econf_recalc,econfs_mean);
    
    %% training
    if(length(state.blocks(state.curBlock).frames)>=state.opts.block_frames)
        
        state.blocks(state.curBlock).econfs = mean(state.blocks(state.curBlock).econfs);
        state.blocks(state.curBlock).minconfs = min(cell2mat({state.blocks(state.cand_blocks).maxminconf})', state.blocks(state.curBlock).econfs')';
        [state.blocks(state.curBlock).maxminconf, ~] = max(state.blocks(state.curBlock).minconfs);
        
        maxes = state.blocks(state.curBlock).minconfs == state.blocks(state.curBlock).maxminconf;
        if strcmp(state.chooseMax, 'linear')
            pid = length(maxes);
        elseif strcmp(state.chooseMax, 'oldest')
            pid = find(maxes, 1, 'first');
        elseif strcmp(state.chooseMax, 'latest')
            pid = find(maxes, 1, 'last');
        elseif strcmp(state.chooseMax, 'stochastic')
            pid = randsample(length(state.blocks(state.curBlock).minconfs),1,true,maxes);
        elseif strcmp(state.chooseMax, 'random')
            pid = randsample(length(state.blocks(state.curBlock).minconfs),1,true);
        else
            error('chooseMax error');
        end
%         fprintf('pid: %d\n', state.cand_blocks(pid));
        state.blocks(state.curBlock).maxminparent = state.cand_blocks(pid);
        state.blocks(state.curBlock).parents  = state.cand_blocks;
        
%        weights = minconfs/sum(minconfs);
%        rel_parents = cand_blocks(weights>=1/opts.parent_blocks);%cand_blocks(weights == value); %cand_blocks; % cand_blocks(weights>=1/opts.parent_blocks);
%        blocks(curBlock).rel_parents = rel_parents;
        
        %             pos_frames = [cell2mat({blocks(rel_parents).frames}), blocks(curBlock).frames];
        if strcmp(state.chooseSample, 'all')
            pos_frames = [state.blocks(state.cand_blocks(pid)).frames, state.blocks(state.curBlock).frames];
        elseif strcmp(state.chooseSample, 'only')
            pos_frames = state.blocks(state.curBlock).frames;
        else
            error('chooseSample error');
        end

%             pos_frames = blocks(curBlock).frames;
        neg_frames = state.blocks(state.curBlock).frames;
        pos_data = cell2mat(state.total_pos_data(pos_frames));
        neg_data = cell2mat(state.total_neg_data(neg_frames));

        [state.net_fc, state.hardnegs] = gnet_finetune_hnm(state.blocks(state.cand_blocks(pid)).cnn,pos_data,neg_data,state.opts,...
            'maxiter',state.opts.maxiter_update,'learningRate',state.opts.learningRate_update);
%                         [net_fc] = gnet_finetune(blocks(cand_blocks(pid)).cnn,pos_data,neg_data,opts,...
%                             'maxiter',opts.maxiter_update,'learningRate',opts.learningRate_update);
        state.blocks(state.curBlock).cnn = state.net_fc;

        state.cand_blocks(end+1) = state.curBlock;
        if(strcmp(state.chooseScoring, 'single') || length(state.cand_blocks)>state.opts.parent_blocks)
            state.blocks(state.cand_blocks(1)).cnn = single([]);
            state.cand_blocks = state.cand_blocks(2:end);
        end

        state.curBlock = state.curBlock+1;
    end    
    
%     %% DISPLAY
%     if display
%         subplot(2,1,1);
%         imshow(img,'initialmagnification','fit'); hold on;
%         
%         rectangle('Position', config.Data.gt(To,:), 'EdgeColor', [0 1 0], 'Linewidth', 3);
%         rectangle('Position', result(To,:), 'EdgeColor', [1 0 0], 'Linewidth', 3);
%         set(gca,'position',[0 0.5 1 0.5]);
%         
%         text(10,10,num2str(To),'Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30); % frameIndex
%         text(10,size(img,1)-10,num2str(score_max),'Color','b', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30); % frameIndex
%         hold off; drawnow; pause(0.001);
%         
%         if(opts.draw)
%             imwrite(frame2im(getframe(gca)),fullfile(config.Dir.draw, sprintf('%03d_samples.jpg', To)));
%         end
%         
%         subplot(2,1,2); plot(1:nFrames,score); grid on; xlim([1,nFrames]); ylim([0,1]);
%         set(gca,'position',[0.1 0.05 0.8 0.4]);
%     end
end

function r = overlap_ratio(rect1, rect2)

    inter_area = rectint(rect1,rect2);
    union_area = rect1(:,3).*rect1(:,4) + rect2(:,3).*rect2(:,4) - inter_area;

    r = inter_area./union_area;
end
