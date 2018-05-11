clear, close all hidden
imgsize=80;
%load deep001.mat
%% Create Datastore

digitData = imageDatastore('img/ds','IncludeSubfolders',true,'LabelSource','foldernames');

%% Mostra alcune immagini random

figure('units','normalized','outerposition',[0 0 1 1]);
perm = randperm(222,60);
for i = 1:60
    subplot(6,10,i);
    imshow(digitData.Files{perm(i)});
end

%% Calcola le immagini per ogni categoria
labelCount = countEachLabel(digitData)

img = readimage(digitData,1);
size(img)
%% Splitta in training data e validation data

max=table2array(labelCount(5,2));
trainNumFiles = round(max*0.8);
[trainDigitData,valDigitData] = splitEachLabel(digitData,trainNumFiles,'randomize');

%% Crea i layers

layers = [
    imageInputLayer([imgsize imgsize 1])

    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(5)
    softmaxLayer
    classificationLayer];

%%
options = trainingOptions('sgdm',...
    'MaxEpochs',3, ...
    'ValidationData',valDigitData,...
    'ValidationFrequency',30,...
    'Verbose',false,...
    'Plots','training-progress',...
    'InitialLearnRate',0.001);

%% Train net
net = trainNetwork(trainDigitData,layers,options);

%% Validate

predictedLabels = classify(net,valDigitData);
valLabels = valDigitData.Labels;

accuracy = sum(predictedLabels == valLabels)/numel(valLabels)

%% Show validated
figure;
perm = randperm(180,20);
for i = 1:20
    subplot(4,5,i);
    imshow(valDigitData.Files{perm(i)});
    title(string(valLabels(perm(i))))
end

%% Try with external images

externalData = imageDatastore('img/save');
externalPredictedLabels = classify(net,externalData);

%% Show external validated
figure;
%perm = randperm(60,20);
for i = 1:16
    subplot(4,4,i);
    imshow(externalData.Files{i});
    title(string(externalPredictedLabels(i)))
end

% for kk=1:4
% I0=imread(['img/save/00' num2str(kk) '.jpg']);
% D1 = classify(net,I0);
% imshow(I0)
% %I0=im2double(I0);
% %I0=rgb2gray(I0);
% 
% %C4 = im2bw(I0,0.4);
%C5 = imresize(C4,[80 80])
%C4 = bwareaopen(I0,50);

%end



