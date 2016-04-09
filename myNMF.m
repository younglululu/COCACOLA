% The code solves one of the following problems: given X, A, and k, find W and H such that 
%    (1) minimize ||X-WH||_F^2+alpha*(sum_(j=1)^n||H(:,j)||_1^2)
%    (2) minimize 1/2*(||X-WH||_F^2+alpha*(sum_(j=1)^n||H(:,j)||_1^2)+beta*tr(HLH'))
%
% <Inputs>
%       X: contig feature matrix (m x n)
%               m : dimension of features (# of samples + tetramer)
%               n : number of contigs
%       weightMat: weight matrix (n x n), expected to be symmetric and sparse matrix 
%       k: estimated cluster number upperbound. Expect k < min(m,n)
%
% (Below are optional arguments: can be set by providing name-value pairs)
%       MODE: 1 to use formulation (1), denoting binning
%             2 to use formulation (2), denoting binning + graph regularization
%             Default is 1
%       ALPHA: Parameter alpha controls the sparsity of columns of H
%              Default is the Lagrange Multiplier approximation mentioned in paper    
%       BETA: Parameter beta only needed in the formulation (2), which controls the belief of graph regularization      
%             Default is 1. No good justfication for this default value, and you might want to try other values  
%       MIN_ITER: Minimum number of iterations. Default is 5.
%       MAX_ITER: Maximum number of iterations. Default is 20.
%       TOL : Stopping tolerance. Default is 1e-3. If you want to obtain a more accurate solution, decrease TOL and increase MAX_ITER at the same time.
%       W_INIT : (m x k) initial value for W.
%       H_INIT : (k x n) initial value for H.
%
% <Outputs> 
%       W: clustering-centroid matrix (m x k)
%       H: clustering-coefficient matrix (k x n)
%       label: contigs-clustering indicator vector (n x 1)
%       
% <Usage Examples>
%       Please refer to example.m
%
%
function [W,H,label]=myNMF(X,weightMat,k,options)
    tic% Dimensions
    [m,n]=size(X); fprintf(' m= %d\n n=%d\n',m,n);
    sigmaHN = min(X(X>0)) * sqrt(pi/2);
    
    % parameter validation check
    if min(X(:))<0, error('Input X should be nonnegative!'); end
    if n<k, error('Observation cannot be less than the number of clusters!'); end

    % Default configuration
    par = [];
    par.MODE = 1;
    par.ALPHA = 0;
    par.BETA = 1;
    par.MIN_ITER = 5;
    par.MAX_ITER = 20;
    par.TOL = 1;
    
    W = rand(m,k);
    H = rand(k,n);

    % Read optional parameters
    if isfield(options,'MODE'), par.MODE = options.MODE; fprintf('MODE is %d \n', par.MODE); end
    if isfield(options,'MIN_ITER'), par.MIN_ITER = options.MIN_ITER; fprintf('MIN_ITER is %d \n', par.MIN_ITER); end
    if isfield(options,'MAX_ITER'), par.MAX_ITER = options.MAX_ITER; fprintf('MAX_ITER is %d \n', par.MAX_ITER); end
    if isfield(options,'TOL'), par.TOL = options.TOL; fprintf('TOL is %f \n', par.TOL); end
    if isfield(options,'ALPHA'), par.ALPHA = options.ALPHA; fprintf('ALPHA is %f \n', par.ALPHA); end
    if isfield(options,'BETA'), par.BETA = options.BETA; fprintf('BETA is %f \n', par.BETA); end
    if isfield(options,'W_INIT'), W = options.W_INIT; disp('Initialize W using input W_INIT'); end
    if isfield(options,'H_INIT'), H = options.H_INIT; disp('Initialize H using input H_INIT'); end
    
    % process weight matrix to the regularized Laplacian
    fprintf('[t=%f] Starting processing weight matrix... \n', toc);
    if size(weightMat,1) ~= n || size(weightMat,2) ~= n, weightMat(n,n) = 0; end
      
    A = weightMat + weightMat';
    Dcol = full(sum(A,2));
    D_mhalf = spdiags(Dcol .^-.5,0,n,n);
    D_mhalf(isinf(D_mhalf)) = 0;
    A = D_mhalf*A*D_mhalf;

    if par.MODE == 2 && par.BETA == 0, par.MODE = 1; end;
    switch par.MODE
        case 1
            par.BETA = 0; fprintf('BETA is set zero in MODE 1 \n');
        case 2
            sbetaI = sqrt(par.BETA).*eye(k);
        otherwise
            error('metaNMF MODE can be only 1 or 2!');
    end
    
    % select proper alpha using the Lagrange Multiplier approximation mentioned in paper
    if ~isfield(options,'ALPHA'), 
        Htmp = nnlsm(W,X,H,'as',sigmaHN);
        yval = W'*(X-W*Htmp);
        xval = Htmp*diag(sum(Htmp)-1);
        alphaOpt = (sum(sum(xval.*yval))-sum(xval(:))*mean(yval(:)))/(sum(sum(xval.*2))-sum(xval(:))*mean(xval(:)));
        alphaOpt = alphaOpt/size(X, 1);
        if isnan(alphaOpt) || isinf(alphaOpt), alphaOpt = mean(X(:)); end
        
        par.ALPHA = alphaOpt;
        fprintf('ALPHA is selected as %f \n', par.ALPHA);
    end
    
    % process mode-related preparation
    if par.ALPHA == 0
        zero1n = []; salphaE = [];
    else
        zero1n = zeros(1,n);
        salphaE = sqrt(par.ALPHA).*ones(1,k);
    end
    
    prevWH = W*H; prevDiffVal = Inf; prevH = H; 
    % starting the optimization   
    fprintf('[t=%f] Starting the optimization... \n', toc);
    for iter=1:par.MAX_ITER
        switch par.MODE
            case 1
                H = nnlsm([W;salphaE],[X;zero1n],H,'as',sigmaHN); 
            case 2
                H = nnlsm([W;salphaE;sbetaI],[X;zero1n;sqrt(par.BETA)*(prevH*A)],H,'as',sigmaHN);
        end
        fprintf('[t=%f] Finished solving H \n', toc);
        prevH = H;
        
        [~,idx] = max(H,[],1); label0 = idx';
        H = spconvert([label0 (1:1:length(label0))' ones(length(label0),1)]);
        if size(H,1) < size(W,2), H(size(W,2),n) = 0; end;
        
        W = nnlsm(H',X',W','bp',sigmaHN); W=W'; fprintf('[t=%f] Finished solving W \n', toc);
        
        currWH = W*H;
        diffVal = sum(sum(abs((prevWH-currWH)))) / 1e4; disp(['diffVal=',num2str(diffVal)])
        if (diffVal <= par.TOL || diffVal == prevDiffVal), break; end
        prevWH = currWH; prevDiffVal = diffVal;
        disp(['Finish iteration: ',num2str(iter)]);        
    end

    [~,idx] = max(H,[],1);
    label = idx';
    
    clear A D_mhalf Dcol;
end

%------------------------------------------------------------------------------------------------------------------------
%                                    Utility Functions 
%------------------------------------------------------------------------------------------------------------------------
function X = nnlsm(A,B,init,solver,sigmaHN)     
    n = size(B,2); kk = size(A,2); 

    X = zeros(size(A,2),size(B,2));
    
    if floor(n/kk) > 0
        XCell = cell(1,floor(n/kk));
        parfor i=1: floor(n/kk)
            idx_sub = (1:1:kk)+(i-1)*kk;
            X_sub = blocknnls( A,B(:, idx_sub),init(:,idx_sub),solver,sigmaHN);       
            XCell{i} = X_sub; 
        end
        X(:, 1:1:floor(n/kk)*kk) = cell2mat(XCell); clear XCell;
    end
    
    if mod(n, kk) > 0
        idx_sub = kk*floor(n/kk)+1:n;
        X_sub = blocknnls( A,B(:, idx_sub),init(:,idx_sub),solver,sigmaHN);
        X(:, idx_sub) = X_sub;
    end
end 