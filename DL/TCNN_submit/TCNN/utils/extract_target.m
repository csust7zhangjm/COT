function [ output_args ] = extract_target( seq, method )
%EXTRACT_TARGET Summary of this function goes here
%   Detailed explanation goes here

drawDir = fullfile('result_target',seq);
if ~exist(drawDir,'dir')
    mkdir(drawDir);
end

conf = genConfig(seq, method);
bbox = load(fullfile(conf.Dir.result,'result.mat'));
bbox = bbox.result;

gt = conf.Data.gt;
nFrames = size(gt,1);

for i=1:nFrames
    
    img = imread(conf.Dir.imgs{i});
    target = img(max(1,bbox(i,2)):min(end,bbox(i,2)+bbox(i,4)),...
        max(1,bbox(i,1)):min(end,bbox(i,1)+bbox(i,3)),:);
    
    imwrite(target,fullfile(drawDir, sprintf('%03d.jpg', i)));
end
