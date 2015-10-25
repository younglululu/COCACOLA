function [Wnew, labelNew, scoreRecordArr] = clustAgg_Lmethod(X, W, label, distance)

[~,n]=size(X);
clusterNum = size(W, 2);

cluster= cell(1,clusterNum);
for i=1:n,cluster{label(i)}(end+1) = i;end

clusterNumArr = [];
scoreRecordArr = [];
WCell = cell(1,clusterNum);
labelCell = cell(1,clusterNum);

[dist,dataClusterMat] = calcPdist(X,W,cluster,distance);
clusterNumArr(end+1) = clusterNum;
WCell{clusterNum} = W;
labelCell{clusterNum} = label;

newLabel = label;
newW = W;

intraDist = sum(min(dataClusterMat));
scoreRecordArr(end+1) = intraDist;

for i=2:round(clusterNum*0.75)
   [MinRow, IdxDow] = min(dist);
   [currDist, MinJ] = min(MinRow); 
   MinI = IdxDow(MinJ);
   
   if MinI > MinJ
      t=MinI;
      MinI=MinJ;
      MinJ=t;
   end
   
   clusterISize = length(cluster{MinI});
   clusterJSize = length(cluster{MinJ});

   newLabel(newLabel == MinJ) = MinI;
   newLabel(newLabel > MinJ) = newLabel(newLabel > MinJ) - 1;
   
   newW(:,MinI) = (clusterISize*newW(:,MinI) + clusterJSize*newW(:,MinJ)) / (clusterISize+clusterJSize);
   newW(:,MinJ) = []; 
   
   cluster{MinI} = [cluster{MinI} cluster{MinJ}];
   cluster(MinJ) = [];
   
   [distCol,dataClusterRow] = updatePdist(X, newW, cluster, MinI, distance); distCol(MinI) = inf;
   dist(:,MinJ) = []; dist(MinJ,:) = [];
   dist(:,MinI) = distCol; dist(MinI,:) = distCol';

   dataClusterMat(MinJ,:) = []; dataClusterMat(MinI,:) = dataClusterRow; intraDist = sum(min(dataClusterMat));
   
   newLabelCopy = newLabel; newWCopy = newW;
   
   scoreRecordArr(end+1) = intraDist;

   clusterNumArr(end+1) = clusterNum - i + 1;
   WCell{clusterNum - i + 1} = newWCopy;
   labelCell{clusterNum - i + 1} = newLabelCopy;
end
scoreRecordArr = real(scoreRecordArr);

fitErrArr = [];
fitErrArr(end+1) = fitLinear(clusterNumArr, scoreRecordArr);

[~, planeFitParam] = fitLinear(clusterNumArr(1:2),[scoreRecordArr(1) scoreRecordArr(1)]);

for i=2:length(scoreRecordArr)-1
    fitErr1 = calFitErr(clusterNumArr(1:i),scoreRecordArr(1:i),planeFitParam);
    [fitErr2, ~] = fitLinear(clusterNumArr(i:end),scoreRecordArr(i:end));
    fitErrArr(end+1) = (i*fitErr1+(length(scoreRecordArr)-i)*fitErr2)/length(scoreRecordArr);
end

fitErrArr(end+1) = calFitErr(clusterNumArr,scoreRecordArr,planeFitParam);
[~,I] = min(fitErrArr); 

if length(I) > 1, I = max(I); end

optI = I;
kNew = clusterNumArr(optI);
Wnew = WCell{kNew};
labelNew = labelCell{kNew};

end


function [fitErr, fitParam] = fitLinear(x,y)
    fitParam=polyfit(x,y,1);
    fitErr = calFitErr(x,y,fitParam);
end

function fitErr = calFitErr(x,y,fitParam)
    estimate = fitParam(1)*x + fitParam(2);
    fitErr=sqrt(sum((y(:)-estimate(:)).^2)/numel(y));
end

function [distCol,dataClusterRow] = updatePdist(X, newW, cluster, MinI, distance)

WCol = newW(:,MinI);
clusterNum = size(newW, 2);
distCol = zeros(clusterNum,1);
switch distance
    case 1
        tmpMat = cell2mat(arrayfun(@(j) sum(abs(bsxfun(@minus,X, WCol(:,j))))',1:1:size(WCol, 2),'UniformOutput',0));
    case 2   
        tmpDist = bsxfun(@plus,dot(WCol,WCol,1)',dot(X,X,1))-2*(WCol'*X); 
        tmpMat = sqrt(tmpDist)';
    otherwise
        error('distance can be only 1 or 2!');    
end

dataClusterRow = tmpMat';

Xsub = X(:,cluster{MinI});
sizeCluster = length(cluster{MinI});

switch distance
    case 1
        tmpMat1 = cell2mat(arrayfun(@(j) sum(sum(abs(bsxfun(@minus,Xsub, newW(:,j))))),1:1:size(newW, 2),'UniformOutput',0));
    case 2   
        tmpMat1 = sum(sqrt(bsxfun(@plus,dot(newW,newW,1)',dot(Xsub,Xsub,1))-2*(newW'*Xsub)), 2); 
    otherwise
        error('distance can be only 1 or 2!');    
end

for j=1: clusterNum
    tmpVal = (mean(tmpMat(cluster{j}))+tmpMat1(j)/sizeCluster)/2;
    distCol(j) = tmpVal;
end
distCol(MinI) = inf;

end


function [distMat,dataClusterMat] = calcPdist(X, W, cluster, distance)

clusterNum = size(W, 2);
distMat = zeros(clusterNum);

switch distance
    case 1
        distCell = cell(1,size(W,2));
        parfor j=1: size(W,2), distCell{j} = sum(abs(bsxfun(@minus,X,W(:,j))))'; end
        tmpMat = cell2mat(distCell);
    case 2   
        tmpDist = bsxfun(@plus,dot(W,W,1)',dot(X,X,1))-2*(W'*X); 
        tmpMat = sqrt(tmpDist)';
    otherwise
        error('distance can be only 1 or 2!');    
end
tmpMat = tmpMat';
dataClusterMat = tmpMat;

for i=1: clusterNum
for j=i+1: clusterNum
    tmpVal = (mean(tmpMat(i, cluster{j})) + mean(tmpMat(j, cluster{i})))/2; 
    distMat(i,j) = tmpVal; distMat(j,i) = tmpVal;
end
end

distMat = distMat + diag(ones(1,size(distMat,2))*inf);
end
