function [X,grad,iter] = blocknnls( A,B,init,solver,sigmaHN)

% disp('size A');size(A)
% disp('size B');size(B)
% disp('size init');size(init)

try
    switch solver
        case 'bp'
            [X,grad,iter] = nnlsm_blockpivot(A,B,0,init);
        case 'as'    
            [X,grad,iter] = nnlsm_activeset(A,B,0,0,init);
    end    
catch ME
    disp('Original blocknnls failed. Try again with random initialization');
    try
        switch solver
            case 'bp'
                [X,grad,iter] = nnlsm_blockpivot(A,B);
            case 'as'    
                [X,grad,iter] = nnlsm_activeset(A,B);
        end    
    catch ME
    
    % ill-conditioned problem, padding A,B with very small random values
    disp('Again failed. Now padding A and B and redo the blocknnls...');
    [~,k]=size(A);
    A1 = [A; abs(normrnd(0,sigmaHN,k,size(A,2)))];
    B1 = [B;abs(normrnd(0,sigmaHN,k,size(B,2)))];

    
    try
        switch solver
            case 'bp'
                [X,grad,iter] = nnlsm_blockpivot(A1,B1,0,init);
            case 'as'    
                [X,grad,iter] = nnlsm_activeset(A1,B1,0,0,init);
        end 
    catch ME
        disp('padding blocknnls also failed. Now switch to the column-wise lsqnonneg...');
        
        X = zeros(size(A,2),size(B,2));
        grad = zeros(size(B));
        iter = 0;

        for i=1: size(B,2)
            [x_sub,~,residual_sub,~,output_sub] = lsqnonneg(A,B(:,i));

            X(:,i) = x_sub;
            grad(:,i) = residual_sub;
            iter = iter + output_sub.iterations;
        end
        grad = -A'*grad;
    end
    
    end
end