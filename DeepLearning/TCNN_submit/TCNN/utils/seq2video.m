function [  ] = seq2video( seqPath, videoPath, framerate )
%SEQ2VIDEO Summary of this function goes here
%   Detailed explanation goes here

seq = parseImg(seqPath);

% if ~exist(videoPath,'dir')
%     mkdir(videoPath);
% end

outputVideo = VideoWriter(videoPath);
outputVideo.FrameRate = framerate;
open(outputVideo);

for ii = 1:length(seq)
    img = imread(seq{ii});
    writeVideo(outputVideo,img);
end

close(outputVideo);

end

