function [ feat ] = sample_features_fcX(net, ims, opts, enableSoftMax)
%SAMPLE_FEATURES_CONVX Summary of this function goes here
%   Detailed explanation goes here


n = size(ims,4);
nBatches = ceil(n/opts.batchSize);

net.layers = net.layers(1:end-1);
if enableSoftMax, net.layers{end+1} = struct('type','softmax'); end

for i=1:nBatches
    
    batch = ims(:,:,:,opts.batchSize*(i-1)+1:min(end,opts.batchSize*i));
    if(opts.useGpu)
        batch = gpuArray(batch);
    end
    
    res = vl_simplenn(net, batch, [], [], ...
        'disableDropout', true, ...
        'conserveMemory', true, ...
        'sync', true) ;
    
    f = gather(res(end).x) ;
    if ~exist('feat','var')
        feat = zeros(size(f,1),size(f,2),size(f,3),n,'single');
    end
    feat(:,:,:,opts.batchSize*(i-1)+1:min(end,opts.batchSize*i)) = f;
    
end