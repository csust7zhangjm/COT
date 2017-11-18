function [ images, region ] = config2inputs( config )
%CONFIG2INPUTS Summary of this function goes here
%   Detailed explanation goes here

images = config.Dir.imgs;
region = config.Data.gt(:,:,1);
region = region(:)';

end

