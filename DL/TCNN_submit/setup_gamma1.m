if(isempty(gcp('nocreate')))
    parpool;
end
addpath '/home/mooyeol/Documents/MATLAB/VOTnew16/TCNN_submit/matconvnet/matlab';
vl_setupnn;
gpuDevice(1);
addpath '/home/mooyeol/Documents/MATLAB/VOTnew16/TCNN_submit/TCNN';
addpath '/home/mooyeol/Documents/MATLAB/VOTnew16/TCNN_submit/TCNN/utils';
