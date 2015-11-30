clc;
clear;
im = imread('sa3.jpg');
im2 = imread('Saudi_license_plate.jpg');

im_gray = rgb2gray(im);
im2_gray = rgb2gray(im2);

points = detectSURFFeatures(im_gray);
points2 = detectSURFFeatures(im2_gray);

[features1, validPoints1] = extractFeatures(im_gray, points);
[features2, validPoints2] = extractFeatures(im2_gray, points2);

indexPairs = matchFeatures(features1, features2);

matchedPoints1 = validPoints1(indexPairs(:, 1), :);
matchedPoints2 = validPoints2(indexPairs(:, 2), :);

figure;
showMatchedFeatures(im, im2, matchedPoints1, matchedPoints2, 'montage');

tform = estimateGeometricTransform(matchedPoints1,...
   matchedPoints2, 'projective');

boxPolygon = [1, 1;...                           % top-left
        size(im, 2), 1;...                 % top-right
        size(im, 2), size(im, 1);... % bottom-right
        1, size(im, 1);...                 % bottom-left
        1, 1];                   % top-left again to close the polygon
%     newBoxPolygon = transformPointsForward(tform, boxPolygon);
%     figure;
% imshow(im2);
% hold on;
% line(newBoxPolygon(:, 1), newBoxPolygon(:, 2), 'Color', 'y');
% title('Detected Box');