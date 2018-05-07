function [net] = gnet_finetune(net,pos_data,neg_data,varargin)
%    
% Train a CNN by SGD, with hard minibatch mining.
% originated from cnn_train() in MatConvNet library.
%
% Hyeonseob Nam, 2015
% namhs09@postech.ac.kr

opts.useGpu = true;
opts.conserveMemory = true ;
opts.sync = true ;

opts.maxiter = 30;
opts.learningRate = 0.001;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;

opts.batchSize = 128;
opts.batch_pos = 32;
opts.batch_neg = 96;

opts = vl_argparse(opts, varargin) ;
% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

for i=1:numel(net.layers)
    if strcmp(net.layers{i}.type,'conv')
        net.layers{i}.filtersMomentum = zeros(size(net.layers{i}.filters), ...
            class(net.layers{i}.filters)) ;
        net.layers{i}.biasesMomentum = zeros(size(net.layers{i}.biases), ...
            class(net.layers{i}.biases)) ; %#ok<*ZEROLIKE>
        
        if opts.useGpu
            net.layers{i}.filtersMomentum = gpuArray(net.layers{i}.filtersMomentum);
            net.layers{i}.biasesMomentum = gpuArray(net.layers{i}.biasesMomentum);
        end
    end
end

%% initilizing
if opts.useGpu
    one = gpuArray(single(1)) ;
else
    one = single(1) ;
end
res = [] ;

n_pos = size(pos_data,4);
n_neg = size(neg_data,4);
train_pos_cnt = 0;
train_neg_cnt = 0;

% extract positive batches
train_pos = [];
remain = opts.batch_pos*opts.maxiter;
while(remain>0)
    if(train_pos_cnt==0)
        train_pos_list = randperm(n_pos)';
    end
    train_pos = cat(1,train_pos,...
        train_pos_list(train_pos_cnt+1:min(end,train_pos_cnt+remain)));
    train_pos_cnt = min(length(train_pos_list),train_pos_cnt+remain);
    train_pos_cnt = mod(train_pos_cnt,length(train_pos_list));
    remain = opts.batch_pos*opts.maxiter-length(train_pos);
end

% extract negative batches
train_neg = [];
remain = opts.batch_neg*opts.maxiter;
while(remain>0)
    if(train_neg_cnt==0)
        train_neg_list = randperm(n_neg)';
    end
    train_neg = cat(1,train_neg,...
        train_neg_list(train_neg_cnt+1:min(end,train_neg_cnt+remain)));
    train_neg_cnt = min(length(train_neg_list),train_neg_cnt+remain);
    train_neg_cnt = mod(train_neg_cnt,length(train_neg_list));
    remain = opts.batch_neg*opts.maxiter-length(train_neg);
end

% learning rate
lr = opts.learningRate ;

%% training on training set
% fprintf('\n');
for t=1:opts.maxiter
    fprintf('training batch %3d of %3d ... ', t, opts.maxiter) ;
    iter_time = tic ;
    
    % ----------------------------------------------------------------------
    % get next image batch and labels
    % ----------------------------------------------------------------------
    batch = cat(4,pos_data(:,:,:,train_pos((t-1)*opts.batch_pos+1:t*opts.batch_pos)),...
        neg_data(:,:,:,train_neg((t-1)*opts.batch_neg+1:t*opts.batch_neg)));
    labels = [2*ones(opts.batch_pos,1,'single');ones(opts.batch_neg,1,'single')];
    if opts.useGpu
        batch = gpuArray(batch) ;
    end
    
    % backprop
    net.layers{end}.class = labels ;
    res = vl_simplenn(net, batch, one, res, ...
        'conserveMemory', opts.conserveMemory, ...
        'disableDropout', true,...
        'sync', opts.sync) ;
    
    % gradient step
    for l=1:numel(net.layers)
        if ~strcmp(net.layers{l}.type, 'conv'), continue ; end
        
        net.layers{l}.filtersMomentum = ...
            opts.momentum * net.layers{l}.filtersMomentum ...
            - (lr * net.layers{l}.filtersLearningRate) * ...
            (opts.weightDecay * net.layers{l}.filtersWeightDecay) * net.layers{l}.filters ...
            - (lr * net.layers{l}.filtersLearningRate) / opts.batchSize * res(l).dzdw{1} ;
        
        net.layers{l}.biasesMomentum = ...
            opts.momentum * net.layers{l}.biasesMomentum ...
            - (lr * net.layers{l}.biasesLearningRate) * ....
            (opts.weightDecay * net.layers{l}.biasesWeightDecay) * net.layers{l}.biases ...
            - (lr * net.layers{l}.biasesLearningRate) / opts.batchSize * res(l).dzdw{2} ;
        
        net.layers{l}.filters = net.layers{l}.filters + net.layers{l}.filtersMomentum ;
        net.layers{l}.biases = net.layers{l}.biases + net.layers{l}.biasesMomentum ;
    end
    
    % print information
    objective = gather(res(end).x)/opts.batchSize ;
    iter_time = toc(iter_time);
    fprintf('objective %.3f, %.2f s\n', objective, iter_time) ;
    
end % next batch