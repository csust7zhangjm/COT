%% RUN_CNT
close all;
clear all;
seqs={ struct('name','woman','path','C:\Matlab_Home\DSST\sequences\woman\imgs\','startFrame',1,'endFrame',597,'nz',8,'ext','jpg','init_rect', [207,117,29,103]) };
%% seqs---结构体，包含：（1）序列名--car4 （2）序列路径--'D:\data_seq\car4\img\' （3）起始帧号：1 （4）终止帧号：659 
%% （5）起始帧的ground truth--[70,51,107,87]（左上点x坐标，左上点y坐标，框宽，框高）
idxSeq=1; % 序列索引号
seq=seqs{idxSeq}; 
%% seq---第idxSeq个序列，即car4序列
seq.len=seq.endFrame-seq.startFrame+1;
%% seq.len=99---总帧数
seq.s_frames=cell(seq.len,1);
%% seq.s_frames=cell(99,1)
nz=strcat('%0',num2str(seq.nz),'d');
%% nz=%04d
for i=1:seq.len % i=1:99
    image_no=seq.startFrame+i-1;
    %% image_no=i
    id=sprintf(nz,image_no);
    %% id=000i
    seq.s_frames{i}=strcat(seq.path,id,'.',seq.ext);
    %% seq.s_frames{i}='D:\data_seq\bird2\img\000i.jpg'---第i帧的路径
end
%% seq.s_frames=｛第1帧的路径；第2帧的路径；....；第99帧的路径｝
bSaveImage=0;
results=run_CNT(seq, bSaveImage);