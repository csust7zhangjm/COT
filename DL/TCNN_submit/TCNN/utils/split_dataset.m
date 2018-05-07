function [ output_args ] = split_dataset(  )
%SPLIT_DATASET Summary of this function goes here
%   Detailed explanation goes here

nSplit = 3;

seqs = importdata('./dataset_otb/seq_list.txt');
nSeq = length(seqs);
nFrames = ones(nSeq, 1);

for i = 1:nSeq
    seq = seqs{i};
    conf = genConfig('otb',seq);
    nFrames(i) = length(conf.Dir.imgs);
end

[nFrames_sort, seq_idx] = sort(nFrames,'descend');
seqSets = cell(1,nSplit);
cnt = zeros(1,nSplit);

for i=1:nSeq
    [~,set_id] = min(cnt);
    seqSets{set_id} = [seqSets{set_id}, seq_idx(i)];
    cnt(set_id) = cnt(set_id)+nFrames_sort(i);
end

for k=1:nSplit
    filename = ['./data_split/seq_list_' num2str(k) '.txt'];
    fp = fopen(filename,'w');
    for i=1:length(seqSets{k})
        fprintf(fp,'%s\n',seqs{seqSets{k}(i)});
    end
    fclose(fp);
end


end

