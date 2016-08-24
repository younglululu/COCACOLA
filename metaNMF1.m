% The code solves one of the following problems: given X, A, and k, find W and H such that 
%    (1) minimize 1/2*(||X-WH||_F^2+alpha*(sum_(j=1)^n||H(:,j)||_1^2))
%    (2) minimize 1/2*(||X-WH||_F^2+alpha*(sum_(j=1)^n||H(:,j)||_1^2)+tr(WVW'))
%    (3) minimize 1/2*(||X-WH||_F^2+alpha*(sum_(j=1)^n||H(:,j)||_1^2)+beta*tr(HLH')+tr(WVW'))
%
% <Inputs>
%       X: contig feature matrix (m x n)
%               m : dimension of features (# of samples + tetramer)
%               n : number of contigs
%       weightMat: weight matrix (n x n), expected to be symmetric and sparse matrix 
%       k: estimated cluster number upperbound. Expect k < min(m,n)
%
% (Below are optional arguments: can be set by providing name-value pairs)
%       MODE: 1 to use formulation (1), denoting basic binning
%             2 to use formulation (2), denoting binning + automatic selecting k
%             3 to use formulation (3), denoting binning + automatic selecting k + graph regularization
%             Default is 3
%       ALPHA: Parameter alpha controls the sparsity of columns of H
%              Default is the average of all elements in X. No good justfication for this default value, and you might want to try other values.
%       BETA: Parameter beta only needed in the formulation (3), which controls the belief of graph regularization      
%             Default is 1. No good justfication for this default value, and you might want to try other values  
%       MIN_ITER: Minimum number of iterations. Default is 10.
%       MAX_ITER: Maximum number of iterations. Default is 50.
%       TOL : Stopping tolerance. Default is 1e-3. If you want to obtain a more accurate solution, decrease TOL and increase MAX_ITER at the same time.
%       W_INIT : (m x k) initial value for W.
%       H_INIT : (k x n) initial value for H.
%
% <Outputs> 
%       W: clustering-centroid matrix (m x k)
%       H: clustering-coefficient matrix (k x n)
%       label: contigs-clustering indicator vector (n x 1)
%       iter: Number of iterations
%       
% <Usage Examples>
%       Please refer to example.m
%
%
function [W,H,label]=metaNMF1(X,weightMat,k,options)
    tic% Dimensions
    [m,n]=size(X); ST_RULE = 1; fprintf(' m= %d\n n=%d\n',m,n);
    sigmaHN = min(X(X>0)) * sqrt(pi/2);
    
    % parameter validation check
    if min(X(:))<0, error('Input X should be nonnegative!'); end
    if n<k, error('Observation cannot be less than the number of clusters!'); end

    % Default configuration
    par = [];
    par.MODE = 3;
    par.NNLS_SOLVER = 'as';
    par.ALPHA = 0;
    par.BETA = 1;
    par.MIN_ITER = 1;
    par.MAX_ITER = 10;
    par.TOL = 1e-4;
    
    W = rand(m,k);
    H = rand(k,n);

    % Read optional parameters
    if isfield(options,'MODE'), par.MODE = options.MODE; fprintf('MODE is %d \n', par.MODE); end
    if isfield(options,'MIN_ITER'), par.MIN_ITER = options.MIN_ITER; fprintf('MIN_ITER is %d \n', par.MIN_ITER); end
    if isfield(options,'MAX_ITER'), par.MAX_ITER = options.MAX_ITER; fprintf('MAX_ITER is %d \n', par.MAX_ITER); end
    if isfield(options,'TOL'), par.TOL = options.TOL; fprintf('TOL is %f \n', par.TOL); end
    if isfield(options,'ALPHA'), par.ALPHA = options.ALPHA; fprintf('ALPHA is %f \n', par.ALPHA); end
    if ~isfield(options,'ALPHA'), par.ALPHA = mean(X(:)); fprintf('ALPHA is %f \n', par.ALPHA); end
    if isfield(options,'BETA'), par.BETA = options.BETA; fprintf('BETA is %f \n', par.BETA); end
    if isfield(options,'W_INIT'), W = options.W_INIT; disp('Initialize W using input W_INIT'); end
    if isfield(options,'H_INIT'), H = options.H_INIT; disp('Initialize H using input H_INIT'); end
    
    if min(m,n)<k
        warning('the program expect k <= min(m,n)');     
%         X = [X; abs(normrnd(0,sigmaHN,k,size(X,2)))];
%         W = [W; abs(normrnd(0,sigmaHN,k,size(W,2)))];
    end
    
    if isfield(options,'NNLS_SOLVER')
        par.NNLS_SOLVER = options.NNLS_SOLVER;
        if ~strcmp(par.NNLS_SOLVER,'bp') && ~strcmp(par.NNLS_SOLVER,'as')
            error('Unrecognized nnls_solver: use ''bp'' or ''as''.');
        end
    end  
    
    if par.MODE == 3 && par.BETA == 0, par.MODE = 2; end;
    
    % process weight matrix to the regularized Laplacian
    fprintf('[t=%f] Starting processing weight matrix... \n', toc);
    if size(weightMat,1) ~= n || size(weightMat,2) ~= n
        weightMat(n,n) = 0;
    end
    
    A = weightMat + weightMat'; 
    defaultBeta = n*full(max(sum(A)))/full(sum(A(A>0))); 
    
    selfDiag = spdiags((full(max(sum(A)))+1)*ones(n,1)-full(sum(A, 2)),0,n,n);
    A = A + selfDiag;
    Dcol = full(sum(A, 2));
    D = spdiags(Dcol,0,n,n);
    D_mhalf = spdiags(Dcol .^-.5,0,n,n);
    D_mhalf(isinf(D_mhalf)) = 0;
    A = D_mhalf*A*D_mhalf;
    D = D_mhalf*D*D_mhalf;
    
    % process mode-related preparation
    if par.ALPHA == 0
        zero1n = []; salphaE = [];
    else
        zero1n = zeros(1,n);
        salphaE = sqrt(par.ALPHA).*ones(1,k);
    end
    
    switch par.MODE
        case 1
            par.BETA = 0; fprintf('BETA is set zero in MODE 1 \n');
            % do nothing
        case 2
            par.BETA = 0; fprintf('BETA is set zero in MODE 2 \n');
            zerokm = zeros(k,m);
        case 3
            if isfield(options,'BETA'), par.BETA = options.BETA; fprintf('BETA is %f \n', par.BETA); end
            if ~isfield(options,'BETA'), par.BETA = defaultBeta; fprintf('BETA is %f \n', par.BETA); end
            sbetaI = sqrt(par.BETA).*eye(k);
            zerokm = zeros(k,m);
        otherwise
            error('metaNMF MODE can be only 1 or 2 or 3!');
    end
    
    initSC = getInitCriterion(ST_RULE,X,W,H,D-A,par);
    
    % starting the optimization   
    fprintf('[t=%f] Starting the optimization... \n', toc);
    for iter=1:par.MAX_ITER
        Wnorm = 1./(sqrt(sum(W.^2))+eps);
        switch par.MODE
            case 1
                [H,gradH] = nnlsm([W;salphaE],[X;zero1n],H,'as',sigmaHN); fprintf('[t=%f] Finished solving H \n', toc);
                [W,gradW] = nnlsm(H',X',W','bp',sigmaHN); W=W'; gradW=gradW'; fprintf('[t=%f] Finished solving W \n', toc);
            case 2
                [H,gradH] = nnlsm([W;salphaE],[X;zero1n],H,'as',sigmaHN); fprintf('[t=%f] Finished solving H \n', toc);
                [W,gradW] = nnlsm([H';diag(Wnorm)],[X';zerokm],W','bp',sigmaHN); W=W'; gradW=gradW'; fprintf('[t=%f] Finished solving W \n', toc);
            case 3
                [H,gradH] = nnlsm([W;salphaE;sbetaI],[X;zero1n;0.5*sqrt(par.BETA)*H*A],H,'as',sigmaHN); fprintf('[t=%f] Finished solving H \n', toc);
                [W,gradW] = nnlsm([H';diag(Wnorm)],[X';zerokm],W','bp',sigmaHN); W=W'; gradW=gradW'; fprintf('[t=%f] Finished solving W \n', toc);
        end
        
        if (iter >= par.MIN_ITER)
            SC = getStopCriterion(ST_RULE,X,W,H,D-A,par,gradW,gradH);
            
            if(isnan(SC/initSC)),  disp('SC/initSC=NAN'); break; end
            disp(['SC/initSC=',num2str(SC/initSC)])
            if (SC/initSC <= par.TOL), break; end
        end
        disp(['Finish iteration: ',num2str(iter)]);        
    end

    [~,idx] = max(H,[],1);
    label = idx';
end

%------------------------------------------------------------------------------------------------------------------------
%                                    Utility Functions 
%------------------------------------------------------------------------------------------------------------------------
function [X,grad,iter] = nnlsm(A,B,init,solver,sigmaHN)
    disp('Now switched to batch mode...'); 
                
    n = size(B,2);
    kk = size(A,2); 
    
    X = zeros(size(A,2),size(B,2));
    grad = zeros(size(X)); %grad = zeros(size(B));
    iter = 0;
    
    if floor(n/kk) > 0
        XCell = cell(1,floor(n/kk));
        gradCell = cell(1,floor(n/kk));
        parfor i=1: floor(n/kk)
            idx_sub = (1:1:kk)+(i-1)*kk;
            [X_sub,grad_sub,iter_sub] = blocknnls( A,B(:, idx_sub),init(:,idx_sub),solver,sigmaHN);       
            XCell{i} = X_sub; gradCell{i} = grad_sub; iter = iter + iter_sub;
        end
        X(:, 1:1:floor(n/kk)*kk) = cell2mat(XCell); clear XCell;
        grad(:, 1:1:floor(n/kk)*kk) = cell2mat(gradCell); clear gradCell;
    end
    
    if mod(n, kk) > 0
        idx_sub = kk*floor(n/kk)+1:n;
        [X_sub,grad_sub,iter_sub] = blocknnls( A,B(:, idx_sub),init(:,idx_sub),solver,sigmaHN);
        
        X(:, idx_sub) = X_sub; grad(:, idx_sub) = grad_sub; iter = iter + iter_sub;
    end
end 

%-------------------------------------------------------------------------------
function retVal = getInitCriterion(stopRule,X,W,H,L,par,gradW,gradH)
% STOPPING_RULE : 1 - Normalized proj. gradient
%                 2 - Proj. gradient
%                 3 - Delta by H. Kim
%                 0 - None (want to stop by MAX_ITER or MAX_TIME)
    if nargin~=8
        [gradW,gradH] = getGradient(X,W,H,L,par);
    end
    [m,k]=size(W); [k,n]=size(H); numAll=(m*k)+(k*n);
    switch stopRule
        case 1
            retVal = norm([gradW; gradH'],'fro')/numAll;
        case 2
            retVal = norm([gradW; gradH'],'fro');
        case 3
            retVal = getStopCriterion(3,X,W,H,gradW,gradH);
        case 0
            retVal = 1;
    end
end

%-------------------------------------------------------------------------------
function retVal = getStopCriterion(stopRule,X,W,H,L,par,gradW,gradH)
% STOPPING_RULE : 1 - Normalized proj. gradient
%                 2 - Proj. gradient
%                 3 - Delta by H. Kim
%                 0 - None (want to stop by MAX_ITER or MAX_TIME)
    if nargin~=8
        [gradW,gradH] = getGradient(X,W,H,L,par);
    end
    switch stopRule
        case 1
            pGradW = gradW(gradW<0|W>0);
            pGradH = gradH(gradH<0|H>0);
            pGrad = [gradW(gradW<0|W>0); gradH(gradH<0|H>0)];
            pGradNorm = norm(pGrad);
            retVal = pGradNorm/length(pGrad);
        case 2
            pGradW = gradW(gradW<0|W>0);
            pGradH = gradH(gradH<0|H>0);
            pGrad = [gradW(gradW<0|W>0); gradH(gradH<0|H>0)];
            retVal = norm(pGrad);
        case 3
            resmat=min(H,gradH); resvec=resmat(:);
            resmat=min(W,gradW); resvec=[resvec; resmat(:)]; 
            deltao=norm(resvec,1); %L1-norm
            num_notconv=length(find(abs(resvec)>0));
            retVal=deltao/num_notconv;
        case 0
            retVal = 1e100;
    end
end

%-------------------------------------------------------------------------------
function [gradW,gradH] = getGradient(X,W,H,L,par)
    k=size(W,2); [m,n]=size(X);
    ahplaI = par.ALPHA*ones(k,k);
    
    Wnorm = sum(W.^2);
    columnSum_toNorm = (Wnorm > 0);
    WV = W;
    WV(:, columnSum_toNorm) = WV(:, columnSum_toNorm) ./ repmat(Wnorm(columnSum_toNorm),m,1);
    
    switch par.MODE
        case 1
            gradW = W*(H*H') - X*H';
            gradH = (W'*W)*H - W'*X + ahplaI*H;
        case 2
            gradW = W*(H*H') - X*H' + WV;
            gradH = (W'*W)*H - W'*X + ahplaI*H;
        case 3
            gradW = W*(H*H') - X*H' + WV;
            gradH = (W'*W)*H - W'*X + ahplaI*H + par.BETA*H*L;
        % test case 4
%         case 4
%             Hnorm = sum(H.^2,2);
%             rowSum_toNorm = (Hnorm > 0);
%             VH = H;
%             VH(rowSum_toNorm, :) = VH(rowSum_toNorm, :) ./ repmat(Hnorm(rowSum_toNorm),1,n);
%             
%             gradW = W*(H*H') - X*H' + WV;
%             gradH = (W'*W)*H - W'*X + ahplaI*H + VH;
    end
end

