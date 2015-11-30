negativeFolder = fullfile('F:','KAU','Semester2','DIP','license plate','NLVP');
trainCascadeObjectDetector('LPR.xml',positiveInstances,negativeFolder,'FalseAlarmRate',0.2,'NumCascadeStages',10);
detector = vision.CascadeObjectDetector('LPR.xml');
img = imread('Saudi_license_plate.jpg');

% clear;
% clc;
% Ix = imread('Saudi_license_plate.jpg');
% %I=Ix;
% I = rgb2gray(Ix);
% figure
% imshow(I);
% figure
% histogram(I)
% 
% figure;
% [BW1,t1] = edge(I,'Canny',0.6);
% 
% % imshow(BW1)
% % t1