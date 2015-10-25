% <Inputs>
%       X: contig feature matrix (m x n)
%               m : dimension of features (# of samples + tetramer)
%               n : number of contigs
function corrMat = calCorrMat(X, minThres)
    n = size(X,2); batch = 1000;
    
    An1=bsxfun(@minus,X,mean(X,1));
    An1=bsxfun(@times,An1,1./sqrt(sum(An1.^2,1)));
    
    corrMat = sparse(n,n);
    if floor(n/batch) > 0
        for i=1: ceil(n/batch)
            idx_sub = (1:1:batch)+(i-1)*batch; 
            if i == ceil(n/batch) && mod(n, batch) > 0, idx_sub = batch*floor(n/batch)+1:n; end
            corr1 = An1(:,idx_sub)'*An1;
            
            for rowIdx = 1: size(corr1,1)
                rowCorr = corr1(rowIdx,:);
                corrIdx = find(rowCorr >= minThres);
                
                if ~isempty(corrIdx)
                   for j=1:length(corrIdx)
                        k = corrIdx(j);
                        realI = (i-1)*batch + rowIdx; 
                        if realI<k
                            corrMat(realI,k) = rowCorr(k);
                        end
                   end
                end
            end
        end        
    end
end