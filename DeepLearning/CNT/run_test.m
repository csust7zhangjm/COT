%% RUN_CNT
close all;
clear all;
seqs={ struct('name','woman','path','C:\Matlab_Home\DSST\sequences\woman\imgs\','startFrame',1,'endFrame',597,'nz',8,'ext','jpg','init_rect', [207,117,29,103]) };
%% seqs---�ṹ�壬��������1��������--car4 ��2������·��--'D:\data_seq\car4\img\' ��3����ʼ֡�ţ�1 ��4����ֹ֡�ţ�659 
%% ��5����ʼ֡��ground truth--[70,51,107,87]�����ϵ�x���꣬���ϵ�y���꣬�����ߣ�
idxSeq=1; % ����������
seq=seqs{idxSeq}; 
%% seq---��idxSeq�����У���car4����
seq.len=seq.endFrame-seq.startFrame+1;
%% seq.len=99---��֡��
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
    %% seq.s_frames{i}='D:\data_seq\bird2\img\000i.jpg'---��i֡��·��
end
%% seq.s_frames=����1֡��·������2֡��·����....����99֡��·����
bSaveImage=0;
results=run_CNT(seq, bSaveImage);