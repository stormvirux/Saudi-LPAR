clear;
clc;
% Read the liscence plate image
Ix = imread('saudi2.jpg');

%Convert the image into binary
I = im2bw(Ix,0.4);
imshow(Ix), figure, imshow(I)

% Convert black to white and vice-versa
I = (I == 0);
figure
imshow(I);

% Divde the image into 2x8
[h,w]=size(I);
vSize = [h w];
nParts = [2 8]; %means divide into 4 parts, 2 horizontal, 2 vertical
%
%figure out the size of "regular" block and the last block
 vRegBlockSize = ceil(vSize ./ nParts);
vLastBlockSize = vSize - vRegBlockSize .* (nParts - 1);
% 
% %put the sizes into a vector
vSplitR = [vRegBlockSize(1)*ones(1,nParts(1)-1), vLastBlockSize(1)];
vSplitC = [vRegBlockSize(2)*ones(1,nParts(2)-1), vLastBlockSize(2)];
% 
vSplitC(1)=vSplitC(1)+13;
vSplitC(2)=vSplitC(2)-8;
vSplitC(3)=vSplitC(3)-2;
vSplitC(4)=vSplitC(4)-3;
% %split the image
C = mat2cell(I, vSplitR, vSplitC);
% 
% %access RGB pixel (x,y) in top left {1,1} block
% %p = C{1,1}(x, y, :);

TI=C{1,1};
CC = bwconncomp(TI);
numPixels = cellfun(@numel,CC.PixelIdxList);
for ii=1:length(numPixels)-1
    [smallest,idx] = min(numPixels);
    TI((CC.PixelIdxList{idx}))=0;
    numPixels(idx)=999999;
end
figure, imshow(TI);
rp = regionprops(TI, 'BoundingBox', 'Area');
area = [rp.Area].';
[~,ind] = max(area);
bboxes = rp(ind).BoundingBox;
%// Step #5
% imshow(TI);
% rectangle('Position', bboxes, 'EdgeColor', 'red');
% ocrtxt = ocr(TI,bboxes);
% [ocrtxt.Text]


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
 xmax = min(xmax, size(TI,2));
 ymax = min(ymax, size(TI,1));

% Show the expanded bounding boxes
 expandedBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1];
 rxx=C{2,3};
 IExpandedBBoxes = rectangle('Position', expandedBBoxes, 'EdgeColor', 'red');%insertShape(TI,'Rectangle',expandedBBoxes,'LineWidth',3);

%figure
%imshow(IExpandedBBoxes)
%title('Expanded Bounding Boxes Text')

%Compute the overlap ratio
overlapRatio = bboxOverlapRatio(expandedBBoxes, expandedBBoxes);

%Set the overlap ratio between a bounding box and itself to zero to
%simplify the graph representation.
n = size(overlapRatio,1);
overlapRatio(1:n+1:n^2) = 0;

%Create the graph
g = graph(overlapRatio);

%Find the connected text regions within the graph
componentIndices = conncomp(g);

%Merge the boxes based on the minimum and maximum dimensions.
xmin = accumarray(componentIndices', xmin, [], @min);
ymin = accumarray(componentIndices', ymin, [], @min);
xmax = accumarray(componentIndices', xmax, [], @max);
ymax = accumarray(componentIndices', ymax, [], @max);

%Compose the merged bounding boxes using the [x y width height] format.
textBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1];

%Remove bounding boxes that only contain one text region
numRegionsInGroup = histcounts(componentIndices);
textBBoxes(numRegionsInGroup == 1, :) = [];

%Show the final text detection result.
%ITextRegion = rectangle('Position', textBBoxes, 'EdgeColor', 'red'); %insertShape(TI, 'Rectangle', textBBoxes,'LineWidth',3);

%%imshow(ITextRegion)
title('Detected Text')
% 
ocrtxt = ocr(TI, textBBoxes);
[ocrtxt.Text]




% numPixels(idx)=0;
% [biggest,idx] = max(numPixels);
% I(CC.PixelIdxList{idx})=0;
% figure, imshow(I);
% numPixels(idx)=0;
% [biggest,idx] = max(numPixels);
% I(CC.PixelIdxList{idx})=0;
% figure, imshow(I);


% [mserRegions] = detectMSERFeatures(C{2,4}, ...
%     'RegionAreaRange',[500 8000],'ThresholdDelta',10);
% 
% figure
% imshow(C{2,4})
% hold on
% plot(mserRegions(7), 'showPixelList', true,'showEllipses',false)
% title('MSER regions')
% hold off
% 
% sz = size(C{2,4});
% pixelIdxList = cellfun(@(xy)sub2ind(sz, xy(:,2), xy(:,1)), ...
%     mserRegions.PixelList, 'UniformOutput', false);
% 
% Next, pack the data into a connected component struct.
% mserConnComp.Connectivity = 8;
% mserConnComp.ImageSize = sz;
% mserConnComp.NumObjects = mserRegions.Count;
% mserConnComp.PixelIdxList = pixelIdxList;
% 
% Use regionprops to measure MSER properties
% mserStats = regionprops(mserConnComp, 'BoundingBox', 'Eccentricity', ...
%     'Solidity', 'Extent', 'Euler', 'Image');
% 
% Compute the aspect ratio using bounding box data.
% bbox = vertcat(mserStats.BoundingBox);
% w = bbox(:,3);
% h = bbox(:,4);
% aspectRatio = w./h;
% 
% Threshold the data to determine which regions to remove. These thresholds
% may need to be tuned for other images.
% filterIdx = aspectRatio' > 5;
% filterIdx = filterIdx | [mserStats.Eccentricity] > .992 ;
% filterIdx = filterIdx | [mserStats.Solidity] < .2;
% filterIdx = filterIdx | [mserStats.Extent] < 0.2 | [mserStats.Extent] > 0.9;
% filterIdx = filterIdx | [mserStats.EulerNumber] < -4;
% 
% Remove regions
% mserStats(filterIdx) = [];
% mserRegions(filterIdx) = [];
% 
% Show remaining regions
% figure
% imshow(C{2,4})
% hold on
% plot(mserRegions, 'showPixelList', true,'showEllipses',false)
% title('After Removing Non-Text Regions Based On Geometric Properties')
% hold off
% 
% Get a binary image of the a region, and pad it to avoid boundary effects
% during the stroke width computation.
% regionImage = mserStats(6).Image;
% regionImage = padarray(regionImage, [1 1]);
% 
% Compute the stroke width image.
% distanceImage = bwdist(~regionImage);
% skeletonImage = bwmorph(regionImage, 'thin', inf);
% 
% strokeWidthImage = distanceImage;
% strokeWidthImage(~skeletonImage) = 0;
% 
% Show the region image alongside the stroke width image.
% figure
% subplot(1,2,1)
% imagesc(regionImage)
% title('Region Image')
% 
% subplot(1,2,2)
% imagesc(strokeWidthImage)
% title('Stroke Width Image')
% 
% Compute the stroke width variation metric
% strokeWidthValues = distanceImage(skeletonImage);
% strokeWidthMetric = std(strokeWidthValues)/mean(strokeWidthValues);
% 
% Threshold the stroke width variation metric
% strokeWidthThreshold = 1;
% strokeWidthFilterIdx = strokeWidthMetric > strokeWidthThreshold;
% 
% Process the remaining regions
% for j = 1:numel(mserStats)
% 
%     regionImage = mserStats(j).Image;
%     regionImage = padarray(regionImage, [1 1], 0);
% 
%     distanceImage = bwdist(~regionImage);
%     skeletonImage = bwmorph(regionImage, 'thin', inf);
% 
%     strokeWidthValues = distanceImage(skeletonImage);
% 
%     strokeWidthMetric = std(strokeWidthValues)/mean(strokeWidthValues);
% 
%     strokeWidthFilterIdx(j) = strokeWidthMetric > strokeWidthThreshold;
% 
% end
% 
% Remove regions based on the stroke width variation
% mserRegions(strokeWidthFilterIdx) = [];
% mserStats(strokeWidthFilterIdx) = [];
% 
% Show remaining regions
% figure
% imshow(C{2,4})
% hold on
% plot(mserRegions, 'showPixelList', true,'showEllipses',false)
% title('After Removing Non-Text Regions Based On Stroke Width Variation')
% hold off
% 
% % Get bounding boxes for all the regions
% bboxes = vertcat(mserStats.BoundingBox);
% 
% % Convert from the [x y width height] bounding box format to the [xmin ymin
% % xmax ymax] format for convenience.
%  xmin = bboxes(:,1);
% ymin = bboxes(:,2);
% xmax = xmin + bboxes(:,3) - 1;
% ymax = ymin + bboxes(:,4) - 1;
% 
% % Expand the bounding boxes by a small amount.
%  expansionAmount = 0.02;
%  xmin = (1-expansionAmount) * xmin;
%  ymin = (1-expansionAmount) * ymin;
%  xmax = (1+expansionAmount) * xmax;
%  ymax = (1+expansionAmount) * ymax;
% 
% % Clip the bounding boxes to be within the image bounds
%  xmin = max(xmin, 1);
%  ymin = max(ymin, 1);
%  xmax = min(xmax, size(C{2,4},2));
%  ymax = min(ymax, size(C{2,4},1));
% 
% % Show the expanded bounding boxes
%  expandedBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1];
%  rxx=C{2,4};
%  IExpandedBBoxes = insertShape(rxx,'Rectangle',expandedBBoxes,'LineWidth',3);
% 
% figure
% imshow(IExpandedBBoxes)
% title('Expanded Bounding Boxes Text')
% 
% Compute the overlap ratio
% overlapRatio = bboxOverlapRatio(expandedBBoxes, expandedBBoxes);
% 
% Set the overlap ratio between a bounding box and itself to zero to
% simplify the graph representation.
% n = size(overlapRatio,1);
% overlapRatio(1:n+1:n^2) = 0;
% 
% Create the graph
% g = graph(overlapRatio);
% 
% Find the connected text regions within the graph
% componentIndices = conncomp(g);
% 
% Merge the boxes based on the minimum and maximum dimensions.
% xmin = accumarray(componentIndices', xmin, [], @min);
% ymin = accumarray(componentIndices', ymin, [], @min);
% xmax = accumarray(componentIndices', xmax, [], @max);
% ymax = accumarray(componentIndices', ymax, [], @max);
% 
% Compose the merged bounding boxes using the [x y width height] format.
% textBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1];
% 
% Remove bounding boxes that only contain one text region
% numRegionsInGroup = histcounts(componentIndices);
% textBBoxes(numRegionsInGroup == 1, :) = [];
% 
% Show the final text detection result.
% ITextRegion = insertShape(C{2,4}, 'Rectangle', textBBoxes,'LineWidth',3);
% 
% figure
% imshow(ITextRegion)
% title('Detected Text')
% 
% ocrtxt = ocr(I, textBBoxes);
% [ocrtxt.Text]