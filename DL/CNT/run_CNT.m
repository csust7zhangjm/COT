function results=run_CNT(seq, res_path, bSaveImage)
%32bit is much faster than 64bit matlab. 2s VS 7s
%*************************************************************
%% Copyright (C) Kaihua Zhang (www.kaihuazhang.net).
%% All rights reserved.
%% Date: 01/2016
%% Codes for paper: Kaihua Zhang et al., Robust Visual Tracking via Convolutional Networks without Training, TIP2016
%%
close all
%rand('state',0);
s_frames = seq.s_frames;

para=paraConfig_CNT(seq.name);

rect=seq.init_rect;
p = [rect(1)+rect(3)/2, rect(2)+rect(4)/2, rect(3), rect(4), 0];
sz = para.psize;
param0 = [p(1), p(2), p(3)/sz(1) , p(5), p(4)/p(3), 0]; %param0 = [px, py, sc, th,ratio,phi];   
param0 = affparam2mat(param0);
opt = para.opt;
opt.psize=para.psize;

n_sample = opt.numsample;
param = [];
param.est = param0';

img_color = imread(s_frames{1});

if size(img_color,3)==3
    img	= double(rgb2gray(img_color));
else
    img	= double(img_color);
end 
 
patchsize = [6 6];% 
patchnum(1) = length(patchsize(1)/2 : 1: (sz(1)-patchsize(1)/2));
patchnum(2) = length(patchsize(2)/2 : 1: (sz(2)-patchsize(2)/2));
Fisize = 100;
[Fio, patcho] = designFilters(img, param0, opt, patchsize, patchnum, Fisize);

%%
neg = sampleNeg(img, param.est', opt.psize, 20, opt, 8);%extract negative samples
FiNeg = zeros(36,Fisize);
for i = 1:size(neg,2)
    FiNeg = FiNeg + affineTrainNeg(reshape(neg(:,i),[32 32]), patchsize, patchnum, Fisize);
end
FiNeg = FiNeg/size(neg,2);
%%
Fii =  bsxfun(@minus,Fio,mean(Fio))-bsxfun(@minus,FiNeg,mean(FiNeg));%Fitlers in Eq.(1)
num=seq.endFrame-seq.startFrame+1;

alpha_p = zeros(Fisize, prod(patchnum), num);
res = zeros(num, 6);
duration = 0;

res(1,:) = param.est';

%%******************************************* Do Tracking *********************************************%%

for f = 2:seq.len
     disp(['# ' num2str(f)]);
    
    img_color = imread(s_frames{f});
    
    if size(img_color,3)==3
        img	= double(rgb2gray(img_color));
    else
        img	= double(img_color);
    end        
           
    [wimgs Y param] = affineSample(img, sz, opt, param);    % draw N candidates with particle filter   
        
    patch = affinePatch(wimgs, patchsize, patchnum);          % obtain M patches for each candidate    
       
    if f==2                                                                    
        xo = bsxfun(@minus,patcho,mean(patcho));
        S = Fii'*xo;%Eq.(1)
        alpha_qq = S;
    end
       
    sim = zeros(1,n_sample);
      
    for i = 1:n_sample
        x = bsxfun(@minus,patch(:,:,i),mean(patch(:,:,i)));
        S = Fii'*x; %Eq.(1)
        alpha_p(:,:,i) = S;        
  
        p = S;
        p = reshape(p, 1, numel(p));        
        p  = p./(sqrt(sum(p.^2))+eps);
        q = alpha_qq;
        q = reshape(q, 1, numel(q));
      
        q = q./(sqrt(sum(q.^2))+eps);
        sim(i) = p*q';
    end      
  
    likelihood = sim;
    [v_max,id_max] = max(likelihood)   
    
    param.est = affparam2mat(param.param(:,id_max));
    res(f,:) = param.est';
    
   %%----------------- Update Scheme ----------------%% 
   
   alp = alpha_p(:,:,id_max);
   alp(abs(p)<median(abs(p))) = 0; %Eq.(3)  
   alpha_qq(abs(p)>median(abs(p))) = 0.95*alpha_qq(abs(p)>median(abs(p)))+0.05*alp(abs(p)>median(abs(p))); %Eq.(4)
   neg = sampleNeg(img, param.est', opt.psize, 20, opt, 8);
   FiNeg = zeros(36,Fisize);
   for i = 1:size(neg,2)
       FiNeg = FiNeg + affineTrainNeg(reshape(neg(:,i),[32 32]), patchsize, patchnum, Fisize);
    end
    FiNeg = FiNeg/size(neg,2);    
    Fii =  bsxfun(@minus,Fio,mean(Fio))-bsxfun(@minus,FiNeg,mean(FiNeg));%Update filters
  %%    
    bSaveImage = 1;
    if bSaveImage
        % display the tracking result in each frame
%         te      = importdata([ 'Datasets\' title '\' 'dataInfo.txt' ]);
%         imageSize = [ te(2) te(1) ];

%         if f == 1
%             figure('position',[ 100 100 imageSize(2) imageSize(1) ]);
%             set(gcf,'DoubleBuffer','on','MenuBar','none');
%         end

%         axes(axes('position', [0 0 1.0 1.0]));
%         imagesc(img_color, [0,1]);
        imshow(img_color);
        numStr = sprintf('#%03d', f);
        text(10,20,numStr,'Color','r', 'FontWeight','bold', 'FontSize',20);

        color = [ 1 0 0 ];
        [ center corners ] = drawbox(para.psize, res(f,:), 'Color', color, 'LineWidth', 2.5);

        axis off;
        drawnow;
    end
end

%%******************************************* Save and Display Tracking Results *********************************************%%
results.type = 'ivtAff';
results.res = res;%each row is a rectangle
results.tmplsize = para.psize;%[width, height]
results.fps=(seq.len-1)/duration;
disp(['fps: ' num2str(results.fps)])
