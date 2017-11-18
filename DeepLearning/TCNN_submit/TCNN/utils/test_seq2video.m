
seq_list = './datasets/seq_list_test.txt';
seqs = importdata(seq_list);

for i=1:length(seqs)
    seqPath = fullfile('result_img',seqs{i});
    videoPath = fullfile('result_video',[seqs{i},'.avi']);
    seq2video(seqPath,videoPath,20);
end