% This script can be used to interactively inspect the results

addpath('/home/mooyeol/Documents/MATLAB/VOTnew16/vot-toolkit'); toolkit_path; % Make sure that VOT toolkit is in the path

[sequences, experiments] = workspace_load();

trackers = tracker_load('TCNN0614original');

workspace_browse(trackers, sequences, experiments);

