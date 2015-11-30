clear;
clc;
fname={'saudiarabia9','sa2','sa3','saudi2014','saudi2009','saudi26ADUx','saudi2','sa5'};
for i=1:length(fname)
    Ix = imread(strcat(fname{i},'.jpg'));
    if numel(size(Ix))>=3
        I = rgb2gray(Ix);
    else
        I=Ix;
    end
    figure
    imshow(I);
    [h,w]=size(I);
    vSize = [h w];
    nParts = [2 8]; %means divide into 4 parts, 2 horizontal, 2 vertical
    % 
    % %figure out the size of "regular" block and the last block
    % 
    vRegBlockSize = ceil(vSize ./ nParts);
    vLastBlockSize = vSize - vRegBlockSize .* (nParts - 1);
    % 
    % %put the sizes into a vector
    vSplitR = [vRegBlockSize(1)*ones(1,nParts(1)-1), vLastBlockSize(1)];
    vSplitC = [vRegBlockSize(2)*ones(1,nParts(2)-1), vLastBlockSize(2)];
    % 
    % %split the image
    vSplitC(1)=vSplitC(1)+13;
    vSplitC(2)=vSplitC(2)-8;
    vSplitC(3)=vSplitC(3)-2;
    vSplitC(4)=vSplitC(4)-3;
    C = mat2cell(I, vSplitR, vSplitC);
    [row, col]=size(C);
    for ii=1:row
        for jj=1:col-1
            imwrite(C{ii,jj},strcat(num2str(ii),num2str(jj),num2str(i),'.jpg'),'jpg');
        end
    end
    
    
end