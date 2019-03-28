clc;
rootFolder = 'train\data';
categories = {'Anger','Disgust','Happy','Neutral','Surprise'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
labelCount = countEachLabel(imds);
img = readimage(imds,2);
size(img)
varSize = 32;
conv1 = convolution2dLayer(5,32,'Padding',2,'BiasLearnRateFactor',2);
conv1.Weights = randn([5 5 3 32])*0.0001;
fc1 = fullyConnectedLayer(64,'BiasLearnRateFactor',2);
fc1.Weights = randn([64 576])*0.1;
fc2 = fullyConnectedLayer(5,'BiasLearnRateFactor',2);
fc2.Weights = randn([5 64])*0.1;

layers = [
    imageInputLayer([varSize varSize 3]);
    conv1;
    maxPooling2dLayer(3,'Stride',2);
    reluLayer();
    convolution2dLayer(5,32,'Padding',2,'BiasLearnRateFactor',2);
   
     reluLayer();
     
     averagePooling2dLayer(3,'Stride',2);

    convolution2dLayer(5,64,'Padding',2,'BiasLearnRateFactor',2);
    

     reluLayer();
     averagePooling2dLayer(3,'Stride',2);
    fc1;
    reluLayer();
    fc2;
    softmaxLayer()
    classificationLayer()];


opts = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'MaxEpochs',10, ...
    'MiniBatchSize', 100, ...
       'Verbose', true);
   
[net, info] = trainNetwork(imds, layers, opts);

rootFolder = 'test\data';
imds_test = imageDatastore(fullfile(rootFolder, categories), ...
    'LabelSource', 'foldernames');

labels = classify(net, imds_test);

ii = randi(500);
im = imread(imds_test.Files{ii});
imshow(im);
if labels(ii) == imds_test.Labels(ii)
   colorText = 'g'; 
else
    colorText = 'r';
end
title(char(labels(ii)),'Color',colorText);



confMat = confusionmat(imds_test.Labels, labels)
confMat = confMat./sum(confMat,2);
confMat = bsxfun(@rdivide,confMat,sum(confMat,2))
mean(diag(confMat))

