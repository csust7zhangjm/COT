function [ output_args ] = extract_all(seq_list)

if nargin<1, seq_list = './datasets/seq_list_test.txt'; end

seqs = importdata(seq_list);
for i = 1:length(seqs)    
    seq = seqs{i};
    fprintf('seq: %s...\n',seq);
    extract_target(seq,'gnet');
end