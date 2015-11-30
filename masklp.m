% Open image
I = imread('saudi2.jpg');

% Convert to grayscale
I = rgb2gray(I);

% Show image
figure(1);
imshow(I)
title('Image with objects')

% mask is the initial contour state
%mask = zeros(size(I));
mask = I < 150;

% Show mask
figure(2);
imshow(mask);
title('Initial contour location')
if false
% bw is a mask of the detected objects
numIter = 2500;
bw = activecontour(I, mask, numIter);

% Show detected objects
figure(3);
imshow(bw);
title('Detected objects')
end
