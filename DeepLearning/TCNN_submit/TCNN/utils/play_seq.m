function [ output_args ] = play_seq( seq, methods, draw )
%PLAY_SEQ Summary of this function goes here
%   Detailed explanation goes here

if nargin<3, draw = false; end
if nargin<2, methods = {}; end

if(draw)
    drawDir = fullfile('result_img',seq);
    if ~exist(drawDir,'dir')
        mkdir(drawDir);
    end
end

if ~isempty(methods)
    nMethods = length(methods);
    bbox = cell(1,nMethods);
    for m=1:nMethods
        conf = genConfig('otb',seq, methods{m});
        bbox{m} = load(fullfile(conf.Dir.result,'result.mat'));
        bbox{m} = bbox{m}.result;
    end
else
    nMethods = 0;
    conf = genConfig(seq);
end

colors = [1 0 0; 0 0 1; 1 0 1; 1 1 0; 0 1 1];
gt = conf.Data.gt;
nFrames = size(gt,1);
score = zeros(nMethods,nFrames);
method_legends = methods;
for m=1:nMethods
    method_legends{m} = make_legend(methods{m});
end
    
for i=1:5:nFrames
    
    img = imread(conf.Dir.imgs{i});
    for m=1:nMethods
        score(m,i) = overlap_ratio(bbox{m}(i,:), gt(i,:));
    end
    
    if (i==1)
        figure(7);
        set(gcf,'Position',[200 100 600 800],'MenuBar','none','ToolBar','none');
    end
    
    subplot(2,1,1);
    imshow(img,'initialmagnification','fit'); hold on;
%     rectangle('Position', gt(i,:), 'EdgeColor', [0 1 0], 'Linewidth', 3);
    for m=1:nMethods
        rectangle('Position', bbox{m}(i,:), 'EdgeColor', colors(m,:), 'Linewidth', 3);
    end
    set(gca,'position',[0 0.5 1 0.5]);
    
    % frame index
    text(10,10,num2str(i),'Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30);
    hold off;
    
    if(draw)
        imwrite(frame2im(getframe(gca)),fullfile(drawDir, sprintf('%03d.jpg', i)));
    end
        
%     subplot(2,1,2); hold on;
%     for m=1:nMethods
%         plot(1:nFrames,score(m,:),'Color',colors(m,:));
%     end
%     legend(method_legends);
%     grid on; xlim([1,nFrames]); ylim([0,1]);
%     set(gca,'position',[0.1 0.05 0.8 0.4]);
%     hold off;
    
    drawnow; pause(0.001);
end


function name = make_legend(name)

for i=1:length(name)
    if strcmp(name(i),'_')
        name(i) = '-';
    end
end


function r = overlap_ratio(rect1, rect2)

inter_area = rectint(rect1,rect2);
union_area = rect1(:,3).*rect1(:,4) + rect2(:,3).*rect2(:,4) - inter_area;

r = inter_area./union_area;