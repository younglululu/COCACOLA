function scoreArr = calInternalIdx(X, W)

scoreArr = []; 

D = bsxfun(@plus,dot(W,W,1)',dot(X,X,1))-2*(W'*X); 
centerDist = bsxfun(@plus,dot(W,W,1)',dot(W,W,1))-2*(W'*W); 
minCenterGap = min(centerDist(centerDist>0));
intraSum = sum(sqrt(min(D)));
[~,label] = min(D);

% calculate TSS index
score1 = (intraSum + sum(centerDist(:)) / (size(W,2)*(size(W,2)-1))) / (minCenterGap + 1/size(W,2)); 
scoreArr(end+1) = score1;

% calculate Davies-Bouldin index
[~,n]=size(X); clusterNum = size(W, 2);
cluster= cell(1,clusterNum); for i=1:n,cluster{label(i)}(end+1) = i;end

SArr = zeros(1,clusterNum); RMat = zeros(clusterNum);
for i=1:clusterNum
    if ~isempty(cluster{i}), SArr(i) = mean(D(i, cluster{i})); end
end

for i=1:clusterNum
    for j=i+1:clusterNum, RMat(i,j) = (SArr(i)+SArr(j))/centerDist(i,j); end
end
RMat = RMat + RMat';
score2 = sum(max(RMat))/clusterNum;
scoreArr(end+1) = score2;

% calculate Silhouette index. The higher the better
dataClusterMatCopy = D;
[aArr,I] = min(D);
for i=1:size(D,2), dataClusterMatCopy(I(i),i)=Inf; end
[bArr,~] = min(dataClusterMatCopy);
    
tmp = (bArr-aArr)./max(bArr,aArr);
silArr = zeros(1,clusterNum);
for i=1:clusterNum, 
	if ~isempty(cluster{i})
        silArr(i) = mean(tmp(cluster{i})); 
    else
        silArr(i) = Inf;
	end
end    
score3 = min(silArr);
scoreArr(end+1) = score3;

end
