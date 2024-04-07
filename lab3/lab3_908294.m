clear all
close all
clc

%% read images
% template
boxImage = imread('./immaginiObjectDetection/elephant.jpg');
% desk
sceneImage = imread('./immaginiObjectDetection/clutteredDesk.jpg');

figure(1), clf, imshow(boxImage)
figure(2), clf, imshow(sceneImage)

tic
%% keypoint detection
boxPoints = detectSIFTFeatures(boxImage);
scenePoints = detectSIFTFeatures(sceneImage);

figure(1), clf
imshow(boxImage), hold on
plot(selectStrongest(boxPoints,100)), hold off

figure(2), clf
imshow(sceneImage), hold on
plot(selectStrongest(scenePoints,100)), hold off

%% keypoint description
[boxFeatures, boxPoints] = extractFeatures(boxImage, boxPoints);
[sceneFeatures, scenePoints] = extractFeatures(sceneImage, scenePoints);

%% feature matching with adjusted parameters
boxPairs = matchFeatures(boxFeatures, sceneFeatures, 'MatchThreshold', 100, 'MaxRatio', 0.2);
matchedBoxPoints = boxPoints(boxPairs(:,1), :);
matchedScenePoints = scenePoints(boxPairs(:,2), :);
showMatchedFeatures(boxImage, sceneImage, matchedBoxPoints, ...
    matchedScenePoints, 'montage');

%
%TEST
%

%coordinate = matchedBoxPoints.Location

% Calcola la distanza euclidea dall'origine per ciascuna coppia di coordinate
%distanze_dall_origine = sqrt(sum(matchedBoxPoints.Location.^2, 2));

% Ottieni gli indici per ordinare le coppie di coordinate in base alla distanza dall'origine
%[~, ordine] = sort(distanze_dall_origine);

% Ordina le coppie di coordinate in base all'ordine calcolato
%coordinate_ordinate_per_distanza = coordinate(ordine, :);

%coord_ordin_y = sortrows(coordinate, 2);
%coord_ordin_y_first = coord_ordin_y(1:10, :)
%coord_ordin_y_last = coord_ordin_y(end-10:end, :)


%% geometric consistency check
[tform, inlierBoxPoints, inlierScenePoints]=...
    estimateGeometricTransform(matchedBoxPoints,...
    matchedScenePoints,'affine');
showMatchedFeatures(boxImage, sceneImage, inlierBoxPoints, ...
    inlierScenePoints, 'montage');

%% bounding box drawing
boxPoly = [1 1;
    size(boxImage,2) 1;
    size(boxImage,2) size(boxImage,1);
    1 size(boxImage,1);
    1 1];

newBoxPoly=transformPointsForward(tform,boxPoly);

figure, clf
imshow(sceneImage), hold on
line(newBoxPoly(:,1),newBoxPoly(:,2),'Color','y')
hold off
toc

%% more precise bounding box
figure, clf
imshow(boxImage)
[x, y] = ginput();

x = [x; x(1)];
y = [y; y(1)];

newBoxPoly = transformPointsForward(tform, [x y]);

% Result
figure, clf
imshow(sceneImage), hold on
line(newBoxPoly(:, 1), newBoxPoly(:, 2), 'Color', 'y')
hold off

toc


