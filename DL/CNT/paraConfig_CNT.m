function para=paraConfig_CNT(title)

para.opt = struct('numsample', 600, 'affsig', [4,4,0.01,0.0,0.00,0]);
para.SC_param.mode = 2;
para.SC_param.lambda = 0.01;
para.SC_param.pos = 'ture'; 

para.patch_size = 16;
para.step_size = 8;

para.psize = [32, 32];
para.normalWidth = 240;
para.normalHeight = 120;



