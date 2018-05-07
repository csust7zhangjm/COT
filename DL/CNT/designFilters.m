function [Fi, patch] = designFilters(img, param0, opt, patchsize, patchnum, Fisize)
% function [Fi patch] =  designFilters(img, param0, opt, patchsize, patchnum, Fisize)
% obtain the dictionary for the SGM

% input --- 
%img: input image
% param0: the initial affine parameters
% opt: initial parameters
% patchsize: the size of each patch
% patchnum: the number of patches in one candidate
% Fisize: the number of filters

% output ---
% Fi: the designed filters with K-means algorithm
% patch: the patches obtained from the first frame (vector)

%*************************************************************
%% Copyright (C) Kaihua Zhang.
%% All rights reserved.
%% Date: 01/2016
image = warpimg(img, param0, opt.psize);

patch = zeros(prod(patchsize), prod(patchnum));

blocksize = size(image);
y = patchsize(1)/2;
x = patchsize(2)/2;

patch_centy = y : 1: (blocksize(1)-y);
patch_centx = x : 1: (blocksize(2)-x);
l =1;
for j = 1: patchnum(1)                   % sliding window
    for k = 1:patchnum(2)
        data = image(patch_centy(j)-y+1 : patch_centy(j)+y, patch_centx(k)-x+1 : patch_centx(k)+x);
        patch(:, l) = reshape(data,numel(data),1);
        l = l+1;
    end
end
cluster_options.maxiters = 10;
cluster_options.verbose  = 0;
Fi = vgg_kmeans(double(patch), Fisize, cluster_options);
