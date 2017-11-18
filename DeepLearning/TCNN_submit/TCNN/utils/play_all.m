function [] = play_all(seq_list)

if nargin<1, seq_list = '../GCNN/dataset_otb/seq_list_old.txt'; end

seqs = importdata(seq_list);
seqs = {'Suv', 'Bird1', 'ClifBar', 'Human3', 'Bolt2'};
% seqs = {'Soccer', 'Ironman', 'Matrix', 'Singer2', 'Skating1', 'Freeman4'};
% seqs = {'Deer', 'Crossing', 'Football1', 'Skiing'};
% seqs = {'FaceOcc1', 'Lemming', 'Dog1', 'Mhyang'};
for i = 5%1:length(seqs)
    seq = seqs{i};
    try
        %              RED, BLUE
        play_seq(seq,{'TCNN_original','TCNN_0313_3'},true);

%         %              RED, BLUE, MAGENTA
%         play_seq(seq,{'TCNN_original_bbP1_1_original','TCNN_original_bbP1_1_bbreg','TCNN_original_bbP1_1_mixed'},true);

%                      RED, BLUE, MAGENTA, YELLOW
%         play_seq(seq,{'TCNN_original_bbP1','TCNN_original_bbP1_1','TCNN_original_bbP1_2','TCNN_original_bbP1_3'},true);
    catch err
        disp(['No ' seq]); 
    end
end