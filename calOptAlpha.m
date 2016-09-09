function [alphaOpt,H2] = calOptAlpha(X, Winit, Hinit)

    X1 = X; W1 = Winit; H1 = Hinit;
    options = [];
    options.MODE = 1;
    options.ALPHA = 0;
    options.W_INIT = W1;
    options.H_INIT = H1;
    options.MIN_ITER = 1;
    options.MAX_ITER = 1;

    [~,H2,~] = metaNMF1(X1,sparse([]),size(W1,2),options); 
    
    y = W1'*(X1-W1*H2);
    %x = H2*diag(sum(H2)-1);
	x = H2 .* repmat(sum(H2)-1, size(H2,1),1);
	
    alphaOpt = (sum(sum(x.*y))-sum(x(:))*mean(y(:)))/(sum(sum(x.*2))-sum(x(:))*mean(x(:)));
    alphaOpt = alphaOpt/size(X, 1);
    if isnan(alphaOpt) || isinf(alphaOpt), alphaOpt = 1e2; end
	clear x y;
end
