%clc;
%clear;
img = imread('C:\opencv\opencv\sources\samples\data\digits.png');
%gray = rgb2gray(img);
figure;
imshow(img);
vSize = [1000 2000];
nParts = [50 100]; %means divide into 4 parts, 2 horizontal, 2 vertical

%figure out the size of "regular" block and the last block

vRegBlockSize = ceil(vSize ./ nParts);
vLastBlockSize = vSize - vRegBlockSize .* (nParts - 1);

%put the sizes into a vector
vSplitR = [vRegBlockSize(1)*ones(1,nParts(1)-1), vLastBlockSize(1)];
vSplitC = [vRegBlockSize(2)*ones(1,nParts(2)-1), vLastBlockSize(2)];

%split the image
C = mat2cell(img, vSplitR, vSplitC);

%access RGB pixel (x,y) in top left {1,1} block
%p = C{1,1}(x, y, :);
digit=0;
counter=1;
index=1;
for i=1:50
    for j=1:50
        img = C{i,j};
        train_data(index,:) = img(:);
        train_label(index)=digit;
        counter=counter+1;
        index=index+1;
        if counter>250
            digit=digit+1;
            counter=1;
        end
    end
end
numTrain = 2500;
sz=[20 20];
train_label=transpose(train_label);
mdl = fitcknn(train_data,train_label);
mdl.NumNeighbors = 1;
[~,score] = resubPredict(mdl);
[X,Y,T,~,OPTROCPT,suby,subnames] = perfcurve(train_label,...
    score(:,9),'8')
rloss = resubLoss(mdl);
cvmdl = crossval(mdl);
kloss = kfoldLoss(cvmdl);
digit=0;
counter=1;
index=1;
for i=1:50
    for j=51:100
        img = C{i,j};
        test_data(index,:) = img(:);
        counter=counter+1;
        index=index+1;
        if counter>250
            digit=digit+1;
            counter=1;
        end
    end
end
final=zeros(2500,1,'double');
for i=1:2500
   final(i)=predict(mdl,test_data(i,:));
end

wrong_count=0;
for i=1:2500
    if final(i)~=train_label(i)
        wrong_count=wrong_count+1;
    end
end

accuracy= ((2500-wrong_count)/2500)*100;
accuracy



