function [ imgList ] = parseImg( loc )
% parse image from given seq name

% image extension :
ext = {'jpg', 'png', 'jpeg'};

% parse image
tmpList = {};
for i=1:length(ext)
   extList = dir(fullfile(loc, ['*', ext{i}]));
   tmpList = {tmpList{:}, extList(:).name};
end

% put prefix path to imgList
for i=1:length(tmpList)
   tmpList{i} = fullfile(loc, tmpList{i});
end

imgList = tmpList;
imgList = sort_nat(imgList);

end

