% Perform k-means clustering in parallel or even GPU
%   X: d x n data matrix
%   k: number of seeds
%
% (Below are optional arguments: can be set by providing name-value pairs)
%   repeat: number of overall repeat. Default is 100.
%           Larger repeat value will get more accurate clustering result
%           but with more runtime cost
%   blockLen: number of repeat in one job. Default is 10.
%   start: method used to choose initial cluster centroid positions. Default is 1.  
%          1 to use random initialization
%          2 to use kmeans++ initialization
%          3 to use 10percent pre-clustering as initialization
%   distance: distance measurement to seperate the data. Default is 2
%          1 to use L1-distance (manhattan distance). Relatively slow, can be speedup by GPU
%          2 to use L2-distance (euclidean distance). Very fast due to vectorization by Michael Chen (sth4nth@gmail.com)
%   
% Partially based on the code from Michael Chen (sth4nth@gmail.com)


function [label,optMean,err] = myKmeansPar(X, k, options)

    par = [];
    par.repeat = 1;
    par.blockLen = 1;
    par.start = 1;
    par.distance = 1;
    
    if isfield(options,'repeat'), par.repeat = options.repeat; end
    if isfield(options,'blockLen'), par.blockLen = options.blockLen; end
    if isfield(options,'start'), par.start = options.start; end
    if isfield(options,'distance'), par.distance = options.distance; end
    
%     fprintf('repeat is %d \n', par.repeat); 
%     fprintf('blockLen is %d \n', par.blockLen); 
%     fprintf('start is %d \n', par.start); 
%     fprintf('distance is %d \n', par.distance); 
     
    blockNum = ceil(par.repeat/par.blockLen);
    candSeedPool = randperm(max(10000, blockNum));
    
    labelCell = cell(1,blockNum);
    optMeanCell = cell(1,blockNum);
    optErrCell = cell(1,blockNum);
    
    parfor i=1: blockNum
        switch par.distance
            case 1
                [label_tmp,optMean_tmp,err_tmp] = myKmeansL1(X,k,par,candSeedPool(i));
            case 2
                [label_tmp,optMean_tmp,err_tmp] = myKmeansL2(X,k,par,candSeedPool(i));
            otherwise
                error('distance can be only 1 or 2 or 3!');
        end
        
        optErrCell{i} = err_tmp;
        labelCell{i} = label_tmp;
        optMeanCell{i} = optMean_tmp;
    end
    
    [~,I]=min(cell2mat(optErrCell));
        
    label = labelCell{I};
    optMean = optMeanCell{I};
    err = optErrCell{I};
end


function [label,optMean,optErr] = myKmeansL2(X,k,par,randSeed)

if par.start==1, rng(randSeed); end 
label = []; optMean = []; optErr = -1; optK = -1; n = size(X,2);

for cnt = 1: par.blockLen
    last = 0; m = [];iter=1;
    switch par.start
        case 1
            m = initSeedsRand(X, k);
        case 2
            m = initSeedsKmeansPP(X, k, par.distance);
        case 3
            m = X(:,randsample(n,k));
    end
    [~,currLabel] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
    
    while any(currLabel ~= last') && iter<=100
        [u,~,currLabel] = unique(currLabel);
        k_init = length(u);
        
        E = sparse(1:n,currLabel,1,n,k_init,n);  % transform currLabel into indicator matrix
        m = X*(E*spdiags(1./sum(E,1)',0,k_init,k_init));    % compute m of each cluster
        last = currLabel;
        [~,currLabel] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1); % assign samples to the nearest centers
        iter = iter+1;
    end
    
    D = bsxfun(@plus,dot(m,m,1)',dot(X,X,1))-2*(m'*X); err = sum(min(D));
    disp(['optErr: ', num2str(optErr),'  currErr: ', num2str(err),' optK: ', num2str(optK),' k: ', num2str(k_init)]);
    if (cnt == 1) || (err < optErr),label = currLabel'; optErr = err; optMean = m; optK = k_init;end
end
end


function [label,optMean,optErr] = myKmeansL1(X,k,par,randSeed)

if par.start==1, rng(randSeed); end 
label = []; optMean = []; optErr = -1; optK = -1; n = size(X,2);
if gpuDeviceCount>0,disp('Now switch to GPU mode...'); gX = gpuArray(X); end;

for cnt = 1: par.blockLen
    last = 0; m = []; iter = 1;
    switch par.start
        case 1
            m = initSeedsRand(X, k);
        case 2
            m = initSeedsKmeansPP(X, k, par.distance);
        case 3
            m = X(:,randsample(n,k));
    end
    
    if gpuDeviceCount > 0
        gm = gpuArray(m);
        dist = cell2mat(arrayfun(@(j) gather(sum(abs(bsxfun(@minus,gX,gm(:,j))))'),1:1:size(m,2),'UniformOutput',0));
    else
        %dist = cell2mat(arrayfun(@(j) sum(abs(bsxfun(@minus,X,m(:,j))))',1:1:size(m,2),'UniformOutput',0));
        distCell = cell(1,size(m,2));
        parfor j=1: size(m,2), distCell{j} = sum(abs(bsxfun(@minus,X,m(:,j))))'; end
        dist = cell2mat(distCell);
    end
    [minVal,currLabel] = min(dist,[],2); err = sum(minVal); 
    
    while any(currLabel ~= last) && iter<=100
        [u,~,currLabel] = unique(currLabel);
        k_init = length(u);
        
        E = sparse(1:n,currLabel,1,n,k_init,n);  % transform currLabel into indicator matrix
        m = X*(E*spdiags(1./sum(E,1)',0,k_init,k_init));    % compute m of each cluster
        last = currLabel;
        if gpuDeviceCount > 0
            gm = gpuArray(m);
            dist = cell2mat(arrayfun(@(j) gather(sum(abs(bsxfun(@minus,gX,gm(:,j))))'),1:1:size(m,2),'UniformOutput',0));
        else
            %dist = cell2mat(arrayfun(@(j) sum(abs(bsxfun(@minus,X,m(:,j))))',1:1:size(m,2),'UniformOutput',0));
            distCell = cell(1,size(m,2));
            parfor j=1: size(m,2), distCell{j} = sum(abs(bsxfun(@minus,X,m(:,j))))'; end
            dist = cell2mat(distCell);
        end
        [minVal,currLabel] = min(dist,[],2); err = sum(minVal); iter = iter+1;
    end
    disp(['optErr: ', num2str(optErr),'  currErr: ', num2str(err),' optK: ', num2str(optK),' k: ', num2str(k_init)]);
    if (cnt == 1) || (err < optErr),label = currLabel; optErr = err; optMean = m; optK = k_init;end
end
end


% Random initialization
function m = initSeedsRand(X, k)
[d,n] = size(X);
label = ceil(k*rand(1,n));
[u,~,currLabel] = unique(label);   % remove empty clusters
k_init = length(u);
E = sparse(1:n,currLabel,1,n,k_init,n);  % transform currLabel into indicator matrix
m = X*(E*spdiags(1./sum(E,1)',0,k_init,k_init)); 
end

% Kmeans++ initialization
function m = initSeedsKmeansPP(X, k, distance)
[d,n] = size(X);
m = zeros(d,k);
v = inf(1,n);
m(:,1) = X(:,ceil(n*rand));
for i = 2:k
    Y = abs(bsxfun(@minus,X,m(:,i-1)));
    switch distance
        case 1
            dd = sum(Y);
        case 2
            dd = sqrt(dot(Y,Y,1));
        otherwise
            error('distance can be only 1 or 2!');
    end
    v = cumsum(min(v,dd));
    m(:,i) = X(:,find(rand < v/v(end),1));
end
end