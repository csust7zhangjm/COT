function [ bb_samples ] = genSamples( function_handle, varargin )
%GENSAMPLES Summary of this function goes here
%   Detailed explanation goes here

switch (function_handle)
    case 'permute_bboxreg'
        bb = varargin{1};
        n = varargin{2};
        opts = varargin{3};
        trans_f = varargin{4};
        scale_range = varargin{5};
%         aspect_range = varargin{6};
        
        h = opts.imgSize(1); w = opts.imgSize(2);
        
        % [center_x center_y width height]
        bb = [bb(1)+bb(3)/2 bb(2)+bb(4)/2, bb(3:4)];
        
        samples = repmat(bb, [n, 1]);
        samples(:,1:2) = samples(:,1:2) + trans_f * repmat(bb(3:4),n,1) .* (rand(n,2)*2-1);
        samples(:,3:4) = samples(:,3:4) .* opts.scale_factor.^(rand(n,2)*4-2);
        samples(:,3:4) = samples(:,3:4) .* repmat(opts.scale_factor.^(rand(n,1)*scale_range),1,2);

%         samples = repmat(bb, [n, 1]);
%         samples(:,1:2) = samples(:,1:2) + trans_f * repmat(bb(3:4),n,1) .* (rand(n,2)*2-1);
%         samples(:,3:4) = samples(:,3:4) .* opts.scale_factor.^...
%             (rand(n,2)*(aspect_range(2)-aspect_range(1))+aspect_range(1));
%         samples(:,3:4) = samples(:,3:4) .* repmat(opts.scale_factor.^...
%             (rand(n,1)*(scale_range(2)-scale_range(1))+scale_range(1)),1,2);
        
        samples(:,3) = max(10,min(w-10,samples(:,3)));
        samples(:,4) = max(10,min(h-10,samples(:,4)));
        
        % [left top width height]
        bb_samples = [samples(:,1)-samples(:,3)/2 samples(:,2)-samples(:,4)/2 samples(:,3:4)];
        bb_samples(:,1) = max(1-bb_samples(:,3)/2,min(w-bb_samples(:,3)/2, bb_samples(:,1)));
        bb_samples(:,2) = max(1-bb_samples(:,4)/2,min(h-bb_samples(:,4)/2, bb_samples(:,2)));
        bb_samples = round(bb_samples);
        
    case 'permute'
        bb = varargin{1};
        n = varargin{2};
        opts = varargin{3};
        type = varargin{4};
        trans_f = varargin{5};
        scale_f = varargin{6};
        
        h = opts.imgSize(1); w = opts.imgSize(2);
        
        % [center_x center_y width height]
        sample = [bb(1)+bb(3)/2 bb(2)+bb(4)/2, bb(3:4)];
        samples = repmat(sample, [n, 1]);
        
%         r = sqrt(prod(bb(3:4)));
        r = round(mean(bb(3:4)));
        switch (type)
            case 'uniform'
                samples(:,1:2) = samples(:,1:2) + trans_f * r * (rand(n,2)*2-1);
                samples(:,3:4) = samples(:,3:4) .* repmat(opts.scale_factor.^(scale_f*(rand(n,1)*2-1)),1,2);
            case 'gaussian'
                samples(:,1:2) = samples(:,1:2) + trans_f * r * max(-1,min(1,0.5*randn(n,2)));
                samples(:,3:4) = samples(:,3:4) .* repmat(opts.scale_factor.^(scale_f*max(-1,min(1,0.5*randn(n,1)))),1,2);
        end
        samples(:,3) = max(10,min(w-10,samples(:,3)));
        samples(:,4) = max(10,min(h-10,samples(:,4)));
        
        % [left top width height]
        bb_samples = [samples(:,1)-samples(:,3)/2 samples(:,2)-samples(:,4)/2 samples(:,3:4)];
        bb_samples(:,1) = max(1-bb_samples(:,3)/2,min(w-bb_samples(:,3)/2, bb_samples(:,1)));
        bb_samples(:,2) = max(1-bb_samples(:,4)/2,min(h-bb_samples(:,4)/2, bb_samples(:,2)));
        bb_samples = round(bb_samples);
        
    case 'slide'
        bb = varargin{1};
        n = varargin{2};
        opts = varargin{3};
        scale_f=varargin{4};
        
        h = opts.imgSize(1); w = opts.imgSize(2);
        
        range = round([bb(3)/2 bb(4)/2 w-bb(3)/2 h-bb(4)/2]);
        stride = round([bb(3)/5 bb(4)/5]);
        
        [dx, dy, ds] = meshgrid(range(1):stride(1):range(3),...
            range(2):stride(2):range(4),...
            -scale_f:scale_f);
        
        windows = [dx(:) dy(:) bb(3)*opts.scale_factor.^ds(:) bb(4)*opts.scale_factor.^ds(:)];
        
        samples = [];
        while(size(samples,1)<n)
            samples = cat(1,samples,...
                windows(randsample(size(windows,1),min(size(windows,1),n-size(samples,1))),:));
        end
        
        samples(:,3) = max(10,min(w-10,samples(:,3)));
        samples(:,4) = max(10,min(h-10,samples(:,4)));
        
        % [left top width height]
        bb_samples = [samples(:,1)-samples(:,3)/2 samples(:,2)-samples(:,4)/2 samples(:,3:4)];
        bb_samples = round(bb_samples);
end
