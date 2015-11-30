clear;
clc;
theno=cell(7);
Ix = imread('saudi2.jpg');
%I=Ix;
I = rgb2gray(Ix);
figure
imshow(I);
% I = im2bw(Ix,0.4);
% imshow(Ix), figure, imshow(I)


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
% 
% %access RGB pixel (x,y) in top left {1,1} block
% %p = C{i,j}(x, y, :);
 for i=2:2
     for j=1:6
[mserRegions] = detectMSERFeatures(C{i,j}, ...
    'RegionAreaRange',[500 8000],'ThresholdDelta',10);

figure
imshow(C{i,j})
hold on
%plot(mserRegions(9), 'showPixelList', true,'showEllipses',false)
title('MSER regions')
hold off

 sz = size(C{i,j});
 pixelIdxList = cellfun(@(xy)sub2ind(sz, xy(:,2), xy(:,1)), ...
     mserRegions.PixelList, 'UniformOutput', false);

% Next, pack the data into a connected component struct.
mserConnComp.Connectivity = 8;
mserConnComp.ImageSize = sz;
mserConnComp.NumObjects = mserRegions.Count;
mserConnComp.PixelIdxList = pixelIdxList;
% 
% % Use regionprops to measure MSER properties
mserStats = regionprops(mserConnComp, 'BoundingBox', 'Eccentricity', ...
    'Solidity', 'Extent', 'Euler', 'Image','Area','Centroid','Extent');

% Compute the aspect ratio using bounding box data.
bbox = vertcat(mserStats.BoundingBox);
w = bbox(:,3);
h = bbox(:,4);
aspectRatio = w./h;

% Threshold the data to determine which regions to remove. These thresholds
% may need to be tuned for other images.
filterIdx = aspectRatio' > 5;
filterIdx = filterIdx | [mserStats.Eccentricity] > .992 ;
filterIdx = filterIdx | [mserStats.Solidity] < .3 | [mserStats.Solidity] > 0.7;
filterIdx = filterIdx | [mserStats.Extent] < 0.3| [mserStats.Extent] > 0.6;
filterIdx = filterIdx | [mserStats.Area] >3000;% 0.3| [mserStats.Extent] > 0.6;
%filterIdx = filterIdx | [mserStats.EulerNumber] < -4;

% Remove regions
mserStats(filterIdx) = [];
mserRegions(filterIdx) = [];

figure
imshow(C{i,j})
hold on
plot(mserRegions, 'showPixelList', true,'showEllipses',false)
title('MSER regions')
hold off

% Show remaining regions
figure
imshow(C{i,j})
hold on
plot(mserRegions, 'showPixelList', true,'showEllipses',false)
title('After Removing Non-Text Regions Based On Geometric Properties')
hold off

% Get a binary image of the a region, and pad it to avoid boundary effects
% during the stroke width computation.
regionImage = mserStats(1).Image;
regionImage = padarray(regionImage, [1 1]);
% 
% Compute the stroke width image.
distanceImage = bwdist(~regionImage);
skeletonImage = bwmorph(regionImage, 'thin', inf);

strokeWidthImage = distanceImage;
strokeWidthImage(~skeletonImage) = 0;

% Show the region image alongside the stroke width image.
figure
subplot(1,2,1)
imagesc(regionImage)
title('Region Image')

subplot(1,2,2)
imagesc(strokeWidthImage)
title('Stroke Width Image')
% 
% Compute the stroke width variation metric
strokeWidthValues = distanceImage(skeletonImage);
strokeWidthMetric = std(strokeWidthValues)/mean(strokeWidthValues);

% Threshold the stroke width variation metric
strokeWidthThreshold = 1;
strokeWidthFilterIdx = strokeWidthMetric > strokeWidthThreshold;

% Process the remaining regions
for j = 1:numel(mserStats)

    regionImage = mserStats(j).Image;
    regionImage = padarray(regionImage, [1 1], 0);

    distanceImage = bwdist(~regionImage);
    skeletonImage = bwmorph(regionImage, 'thin', inf);

    strokeWidthValues = distanceImage(skeletonImage);

    strokeWidthMetric = std(strokeWidthValues)/mean(strokeWidthValues);

    strokeWidthFilterIdx(j) = strokeWidthMetric > strokeWidthThreshold;

end
% 
% Remove regions based on the stroke width variation
mserRegions(strokeWidthFilterIdx) = [];
mserStats(strokeWidthFilterIdx) = [];

% Show remaining regions
figure
imshow(C{i,j})
hold on
plot(mserRegions, 'showPixelList', true,'showEllipses',false)
title('After Removing Non-Text Regions Based On Stroke Width Variation')
hold off

% Get bounding boxes for all the regions
bboxes = vertcat(mserStats.BoundingBox);

% Convert from the [x y width height] bounding box format to the [xmin ymin
% xmax ymax] format for convenience.
 xmin = bboxes(:,1);
ymin = bboxes(:,2);
xmax = xmin + bboxes(:,3) - 1;
ymax = ymin + bboxes(:,4) - 1;

% Expand the bounding boxes by a small amount.
 expansionAmount = 0.02;
 xmin = (1-expansionAmount) * xmin;
 ymin = (1-expansionAmount) * ymin;
 xmax = (1+expansionAmount) * xmax;
 ymax = (1+expansionAmount) * ymax;

% Clip the bounding boxes to be within the image bounds
 xmin = max(xmin, 1);
 ymin = max(ymin, 1);
 xmax = min(xmax, size(C{i,j},2));
 ymax = min(ymax, size(C{i,j},1));

% Show the expanded bounding boxes
 expandedBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1];
 rxx=C{i,j};
 IExpandedBBoxes = insertShape(rxx,'Rectangle',expandedBBoxes,'LineWidth',3);

figure
imshow(IExpandedBBoxes)
title('Expanded Bounding Boxes Text')

% Compute the overlap ratio
overlapRatio = bboxOverlapRatio(expandedBBoxes, expandedBBoxes);
% 
% Set the overlap ratio between a bounding box and itself to zero to
% simplify the graph representation.
n = size(overlapRatio,1);
overlapRatio(1:n+1:n^2) = 0;

% Create the graph
g = graph(overlapRatio);

% Find the connected text regions within the graph
componentIndices = conncomp(g);

% Merge the boxes based on the minimum and maximum dimensions.
xmin = accumarray(componentIndices', xmin, [], @min);
ymin = accumarray(componentIndices', ymin, [], @min);
xmax = accumarray(componentIndices', xmax, [], @max);
ymax = accumarray(componentIndices', ymax, [], @max);

% Compose the merged bounding boxes using the [x y width height] format.
textBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1];

% Remove bounding boxes that only contain one text region
numRegionsInGroup = histcounts(componentIndices);
textBBoxes(numRegionsInGroup == 1, :) = [];

% Show the final text detection result.
ITextRegion = insertShape(C{i,j}, 'Rectangle', textBBoxes,'LineWidth',3);

figure
imshow(ITextRegion)
title('Detected Text')
marker = imerode(ITextRegion, strel('line',10,0));
Iclean = imreconstruct(marker, ITextRegion);

th = graythresh(Iclean );
BW = im2bw(Iclean , th);

figure;
imshowpair(Iclean , BW, 'montage');
BW = imcomplement(BW);
figure;
imshowpair(Iclean , BW, 'montage');

rootFolder = fullfile('F:','KAU','Semester2','DIP','license plate');
imgSets = [ imageSet(fullfile(rootFolder, '0')), ...
            imageSet(fullfile(rootFolder, '1')), ...
            imageSet(fullfile(rootFolder, '2')), ...
            imageSet(fullfile(rootFolder, '3')), ...
            imageSet(fullfile(rootFolder, '4')), ...
            imageSet(fullfile(rootFolder, '5')), ...
            imageSet(fullfile(rootFolder, '6')), ...
            imageSet(fullfile(rootFolder, '7')), ...
            imageSet(fullfile(rootFolder, '8')), ...
            imageSet(fullfile(rootFolder, 'A')), ...
            imageSet(fullfile(rootFolder, 'B')), ...
            imageSet(fullfile(rootFolder, 'D')), ...
            imageSet(fullfile(rootFolder, 'G')), ...
            imageSet(fullfile(rootFolder, 'J')), ...
            imageSet(fullfile(rootFolder, 'K')), ...
            imageSet(fullfile(rootFolder, 'U')), ...
            imageSet(fullfile(rootFolder, 'X')), ...
            imageSet(fullfile(rootFolder, '9')) ];
[trainingSets] = partition(imgSets, 1.0, 'randomize');
bag = bagOfFeatures(trainingSets);

img1 = read(imgSets(1), 1);
featureVector1 = encode(bag, img1);
img2 = read(imgSets(2), 1);
featureVector2 = encode(bag, img2);
% Plot the histogram of visual word occurrences
figure
bar(featureVector1)
title('Visual word occurrences')
xlabel('Visual word index')
ylabel('Frequency of occurrence')
figure
bar(featureVector2)
title('Visual word occurrences')
xlabel('Visual word index')
ylabel('Frequency of occurrence')
categoryClassifier = trainImageCategoryClassifier(trainingSets, bag);
confMatrix_1 = evaluate(categoryClassifier, trainingSets)
%confMatrix_2 = evaluate(categoryClassifier, validationSets);
mean(diag(confMatrix_1))
% img = imread(fullfile(rootFolder, 'two', '430.png'));
[labelIdx, scores] = predict(categoryClassifier, C{i,j});
img1 = read(imgSets(1), 1);
featureVector1 = encode(bag, img1);
img2 = read(imgSets(2), 1);
featureVector2 = encode(bag, img2);
% Plot the histogram of visual word occurrences
% figure
% bar(featureVector1)
% title('Visual word occurrences')
% xlabel('Visual word index')
% ylabel('Frequency of occurrence')
% figure
% bar(featureVector2)
% title('Visual word occurrences')
% xlabel('Visual word index')
% ylabel('Frequency of occurrence')


% Display the string label
theno{i,j}=categoryClassifier.Labels(labelIdx);
% [labelIdx, scores]=predict(categoryClassifier, validationSets);
wrong_count=0;
% for i=1:2500
%     if answer(i)~=train_label(i)
%         wrong_count=wrong_count+1;
%     end
% end

% 
% ocrtxt = ocr(BW, 'CharacterSet', 'ABCDEFGHIXY','TextLayout', 'Block');
% [ocrtxt.Text]

     end
 end
 bagow(theno)