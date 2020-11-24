


close all;
clear all;
clc;

%% InceptionV3
% Save results to \Results_20200818\resnet50_testall
% No aug
% Batch size 64, max epoch 30, patience 10
%% Load Data
%% Load Data
results_folder = 'E:\OneDrive - The University Of Hong Kong\Postdoc\5_Paper\Results_aug_20200918\inceptionv3';
if ~exist(results_folder, 'dir')
    mkdir(results_folder);
end
imdsTrain =  imageDatastore('E:\OneDrive - The University Of Hong Kong\Postdoc\5_Paper\ImageData_strain_20200901_crop_allaug\TrainData', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
imdsValidation = imageDatastore('E:\OneDrive - The University Of Hong Kong\Postdoc\5_Paper\ImageData_strain_20200901_crop_allaug\TestData', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
%% Load Pretrained Network
 net = inceptionv3();
% net = resnet50;
inputSize = net.Layers(1).InputSize;
% analyzeNetwork(net)
%% Replace Final Layers
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
[learnableLayer,classLayer] 
%% 
numClasses = numel(categories(imdsTrain.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
%% 
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);
%% 
% To check that the new layers are connected correctly, plot the new layer 
% graph and zoom in on the last layers of the network.

% figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
% plot(lgraph)
% ylim([0,10])
%% Freeze Initial Layers
layers = lgraph.Layers;
connections = lgraph.Connections;
% layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);
%% Train Network
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
%% Training Options
options = trainingOptions('adam', ...
    'MiniBatchSize',64 , ...
    'MaxEpochs',100, ...
    'ValidationPatience',10,...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');
%% Training Progress
%  net = trainNetwork(augimdsTrain,lgraph,options);
[net, info] = trainNetwork(augimdsTrain,lgraph,options);
%% Save Training Progress
h= findall(groot,'Type','Figure');
 saveas(h,'training_process.fig');
 saveas(h,'training_process.png');
 saveas(h,string(results_folder)+'\'+'training_process.fig');
 saveas(h,string(results_folder)+'\'+'training_process.png');
%% Classify Validation Images
[YPred,probs] = classify(net,augimdsValidation);
figure;
set(gcf, 'Position',  [10, 10, 1500, 1400])
cm=confusionchart(imdsValidation.Labels,YPred,'ColumnSummary','column-normalized',...
'RowSummary','row-normalized')
cm.Title = 'Fungi Classification Confusion Matrix';
saveas(cm,string(results_folder)+'\'+'Confusion Matrix.fig');
saveas(cm,string(results_folder)+'\'+'Confusion Matrix.png');
%% Confusion Matrix
baseFileName = 'ConfusionMatrix.xlsx';
fullFileName = fullfile(results_folder, baseFileName);
ClassLabels=cellstr(cm.ClassLabels);
xlswrite(fullFileName ,ClassLabels,'ConfusionMatrix_label');
xlswrite(fullFileName ,cm.NormalizedValues,'ConfusionMatrix_value');
%% Ambiguity of Classifications
earlyLayerName = "max_pooling2d_1";
finalConvLayerName = "avg_pool";
% softmaxLayerName = "fc1000_softmax";results_folder, pixelRange, options, imageAugmenter
softmaxLayerName = "predictions_softmax";
pool1Activations = activations(net,...
    augimdsValidation,earlyLayerName,"OutputAs","rows");
finalConvActivations = activations(net,...
    augimdsValidation,finalConvLayerName,"OutputAs","rows");
softmaxActivations = activations(net,...
    augimdsValidation,softmaxLayerName,"OutputAs","rows");

[R,RI] = maxk(softmaxActivations,2,2);
ambiguity = R(:,2)./R(:,1);
[ambiguity,ambiguityIdx] = sort(ambiguity,"descend");

% delete the folder path
for i=1 : length(ambiguity) 
TestImageName {i} = augimdsValidation.Files{i}(strfind ( augimdsValidation.Files{i}, 'TestData\' )+9:end);
end
TestImageName = TestImageName';

for i =1:length(ambiguity)  
LikeliestPoss(i) = R(ambiguityIdx(i),1);
SecondPoss(i) =  R(ambiguityIdx(i),2);
TestImageNameIdx (i) = TestImageName (ambiguityIdx(i));
end
TestImageNameIdx = TestImageNameIdx';

classList = unique(imdsValidation.Labels);

top10Idx = ambiguityIdx;
top10Ambiguity = ambiguity;
mostLikely = classList(RI(ambiguityIdx,1));
secondLikely = classList(RI(ambiguityIdx,2));
LikeliestPoss=LikeliestPoss';
SecondPoss=SecondPoss';

result_ambiguity=table(top10Idx,TestImageNameIdx,top10Ambiguity,mostLikely,LikeliestPoss,secondLikely,SecondPoss,imdsValidation.Labels(ambiguityIdx),...
    'VariableNames',["Image","ImageName","Ambiguity","Likeliest","Likeliest Possibility","Second","Second Possibility","True Class"]);

% Save Ambiguity Results
writetable(result_ambiguity,string(results_folder)+'\'+'Ambiguity List.csv','Delimiter',',','QuoteStrings',true)
%% Explore Observations in t-SNE Plot
softmaxtsne = tsne(softmaxActivations);
numClasses = length(classList);

% color set
for i =1:length(classList)
    colors(i,1)=rand(1);
    colors(i,2)=rand(1);
    colors(i,3)=rand(1);
end

meas=softmaxActivations;
% colors1=colors;
colors1 = [

0	0	0
0	0	0.3
0	0	0.5
0	0	0.9
0	0.4	0.1
0	0.4	0.5
0.4	0.8	0.3
0.4	0.8	0.4
0.4	0.8	0.5
0.4	0.8	0.7
0	0.2	1
0.3	0.1	0.1
0.3	0.1	0.5
0.4	0.3	0
0.4	0.8	0.9
0.4	0.8	1
0.5	0	0
0.5	0	0.4
0.5	0	0.6
0.5	0	0.9
0.7	0.9	0.9
0.7	0.9	1
0.8	0.4	0
0.8	0.4	0.4
1	0.4	0.1
1	0.4	0.4
1	1	0
1	0.5	0.5

];

% figure
Y100 = tsne(meas,'Perplexity',100);
figure
set(gcf, 'Position',  [10, 10, 1500, 1400])
gscatter(Y100(:,1),Y100(:,2),imdsValidation.Labels,colors1);
title('Perplexity 100')
saveas(gcf,string(results_folder)+'\'+'Perplexity 100.fig')
saveas(gcf,string(results_folder)+'\'+'Perplexity 100.png')

% figure
Y50 = tsne(meas,'Perplexity',50);
figure
set(gcf, 'Position',  [10, 10, 1500, 1400])
gscatter(Y50(:,1),Y50(:,2),imdsValidation.Labels,colors1);
title('Perplexity 50')
saveas(gcf,string(results_folder)+'\'+'Perplexity 50.fig')
saveas(gcf,string(results_folder)+'\'+'Perplexity 50.png')

% figure
Y40 = tsne(meas,'Perplexity',40);
figure
set(gcf, 'Position',  [10, 10, 1500, 1400])
gscatter(Y40(:,1),Y40(:,2),imdsValidation.Labels,colors1);
title('Perplexity 40')
saveas(gcf,string(results_folder)+'\'+'Perplexity 40.fig')
saveas(gcf,string(results_folder)+'\'+'Perplexity 40.png')

Y20 = tsne(meas,'Perplexity',20);
figure
set(gcf, 'Position',  [10, 10, 1500, 1400])
gscatter(Y20(:,1),Y20(:,2),imdsValidation.Labels,colors1);
title('Perplexity 20')
saveas(gcf,string(results_folder)+'\'+'Perplexity 20.fig')
saveas(gcf,string(results_folder)+'\'+'Perplexity 20.png')

Y10 = tsne(meas,'Perplexity',10);
figure
set(gcf, 'Position',  [10, 10, 1500, 1400])
gscatter(Y10(:,1),Y10(:,2),imdsValidation.Labels,colors1);
title('Perplexity 10')
saveas(gcf,string(results_folder)+'\'+'Perplexity 10.fig')
saveas(gcf,string(results_folder)+'\'+'Perplexity 10.png')

Y5 = tsne(meas,'Perplexity',5);
figure
set(gcf, 'Position',  [10, 10, 1500, 1400])
gscatter(Y5(:,1),Y5(:,2),imdsValidation.Labels,colors1);
title('Perplexity 5')
saveas(gcf,string(results_folder)+'\'+'Perplexity 5.fig')
saveas(gcf,string(results_folder)+'\'+'Perplexity 5.png')

Y2 = tsne(meas,'Perplexity',2);
figure
set(gcf, 'Position',  [10, 10, 1500, 1400])
gscatter(Y2(:,1),Y2(:,2),imdsValidation.Labels,colors1);
title('Perplexity 2')
saveas(gcf,string(results_folder)+'\'+'Perplexity 2.fig')
saveas(gcf,string(results_folder)+'\'+'Perplexity 2.png')



%%
%%
% doLegend = 'off';
% markerSize = 7;
imdsValidation.Labels(168:171)= 'AmoenusNRRL226';
colors1 = [

0	0	0
0	0	0.3
0	0	0.5
0	0	0.9
0	0.4	0.1
0	0.4	0.5
0.4	0.8	0.3
0.4	0.8	0.4
0.4	0.8	0.5
0.4	0.8	0.7
0	0.2	1
0.3	0.1	0.1
0.3	0.1	0.5
0.4	0.3	0
0.4	0.8	0.9
0.4	0.8	1
0.5	0	0
0.5	0	0.4
0.5	0	0.6
0.5	0	0.9
0.7	0.9	0.9
0.7	0.9	1
0.8	0.4	0
0.8	0.4	0.4
1	0.4	0.1
1	0.4	0.4
1	1	0
1	0.5	0.5

];

pool1tsne = tsne(pool1Activations);
finalConvtsne = tsne(finalConvActivations);
softmaxtsne = tsne(softmaxActivations);

figure;
% subplot(1,3,1);
% gscatter(pool1tsne(:,1),pool1tsne(:,2),imdsValidation.Labels, ...
%     [],'.',markerSize,doLegend);
    
set(gcf, 'Position',  [10, 10, 1500, 1400])
gscatter(pool1tsne(:,1),pool1tsne(:,2),imdsValidation.Labels, colors1);
title("Max pooling activations");
saveas(gcf,string(results_folder)+'\'+'Max pooling activations.fig')
saveas(gcf,string(results_folder)+'\'+'Max pooling activations.png')


figure;
set(gcf, 'Position',  [10, 10, 1500, 1400])
gscatter(finalConvtsne(:,1),finalConvtsne(:,2),imdsValidation.Labels, colors1);
% gscatter(finalConvtsne(:,1),finalConvtsne(:,2),imdsValidation.Labels, ...
%     [],'.',markerSize,doLegend);
title("Final conv activations");
saveas(gcf,string(results_folder)+'\'+'Final conv activations.fig')
saveas(gcf,string(results_folder)+'\'+'Final conv activations.png')


figure;
set(gcf, 'Position',  [10, 10, 1500, 1400])
gscatter(softmaxtsne(:,1),softmaxtsne(:,2),imdsValidation.Labels, colors1);
% gscatter(softmaxtsne(:,1),softmaxtsne(:,2),imdsValidation.Labels, ...
%     [],'.',markerSize,doLegend);
title("Softmax activations");
saveas(gcf,string(results_folder)+'\'+'Softmax activations.fig')
saveas(gcf,string(results_folder)+'\'+'Softmax activations.png')









%% 
% _Copyright 2018 The MathWorks, Inc