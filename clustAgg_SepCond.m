function [newW, newLabel] = clustAgg_SepCond(X, W, label, maxSepCondThres)

maxIter = 20;
[m,n]=size(X);
clusterNum = size(W, 2);

cluster= cell(1,clusterNum);
for i=1:n,cluster{label(i)}(end+1) = i;end

newW = W;
newLabel = label;


for iter = 1: maxIter
    [MinI,MinJ,MinISep,MinJSep,conductance] = getMembershipMat(X,newW,cluster,newLabel);
    if conductance < maxSepCondThres, break; end

    if MinISep > MinJSep
        tmpCluster = cluster{MinI}; cluster{MinI} = cluster{MinJ}; cluster{MinJ} = tmpCluster;
        tmpWCol = newW(:,MinI); newW(:,MinI) = newW(:,MinJ); newW(:,MinJ) = tmpWCol;
        newLabel(newLabel == MinI) = -1; newLabel(newLabel == MinJ) = MinI; newLabel(newLabel == -1) = MinJ;
    end
       
    clusterISize = length(cluster{MinI});
    clusterJSize = length(cluster{MinJ});

    newLabel(newLabel == MinJ) = MinI;
    newLabel(newLabel > MinJ) = newLabel(newLabel > MinJ) - 1;
   
    newW(:,MinI) = (clusterISize*newW(:,MinI) + clusterJSize*newW(:,MinJ)) / (clusterISize+clusterJSize);
    newW(:,MinJ) = []; 

    relatedIdxSet = [cluster{MinJ}];
    Xsub = X(:, relatedIdxSet);

    distCell = cell(1,size(newW,2));
    parfor j=1: size(newW,2), distCell{j} = sum(abs(bsxfun(@minus,Xsub,newW(:,j))))'; end
    dist = cell2mat(distCell);
    [~,subLabel] = min(dist,[],2);

    cluster(MinJ) = [];
    for j=1: length(relatedIdxSet)
        cluster{subLabel(j)}(end+1) = relatedIdxSet(j);
        newLabel(relatedIdxSet(j)) = subLabel(j);
    end

    E = sparse(1:n,newLabel,1,n,size(newW,2),n);
    newW = X*(E*spdiags(1./sum(E,1)',0,size(newW,2),size(newW,2)));
end

end

function [MinI,MinJ,MinISep,MinJSep,conductance] = getMembershipMat(X,W,cluster,label)
    distCell = cell(1,size(W,2));
    parfor j=1: size(W,2), distCell{j} = sum(abs(bsxfun(@minus,X,W(:,j))))'; end
    dist = cell2mat(distCell);

    radius = zeros(1,size(W,2));
    sep = zeros(1,size(W,2));
    overlapCnt = zeros(size(W,2));
    
    for i=1:size(W,2)
        % radius(i) = median(dist(cluster{i},i));
        radius(i) = prctile(dist(cluster{i},i),75);
        
        ambIdx = setdiff(find(dist(:,i)<=radius(i)), cluster{i});
        subLabel = label(ambIdx);
        
        if ~isempty(subLabel)
            for j=1:length(subLabel)
                overlapCnt(i,subLabel(j)) = overlapCnt(i,subLabel(j))+1;
                sep(i) = sep(i) + 1;
                sep(subLabel(j)) = sep(subLabel(j)) + 1;
            end            
        end
    end
    
    overlapCnt = overlapCnt+overlapCnt';
    overlapCnt(find(tril(overlapCnt))) = -1;
    
    for i=1:size(W,2)
        for j=i+1:size(W,2)
            overlapCnt(i,j) = overlapCnt(i,j)/min(length(cluster{i}),length(cluster{j}));           
        end
        sep(i) = sep(i)/length(cluster{i});
    end

    [rowIndice,colIndice] = find(overlapCnt > 0);
    if isempty(rowIndice)
        MinI = -1; MinJ = -1; conductance = -1; MinISep = -1; MinJSep = -1;
    else
        result = zeros(length(rowIndice),3);
        for i=1:length(rowIndice)
            result(i,:) = [rowIndice(i) colIndice(i) overlapCnt(rowIndice(i),colIndice(i))];
        end

        [maxConduct, idx] = max(result(:,3));

        MinI = result(idx,1); MinJ = result(idx,2);
        MinISep = sep(MinI); MinJSep = sep(MinJ); 
        conductance = result(idx,3);
    end
    
end

