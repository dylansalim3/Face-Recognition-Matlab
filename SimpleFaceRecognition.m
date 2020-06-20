% n is the number of subjects
n = 4;
% You can press stop button manually on tranining plot(on top right corner besides number of iterations) once accuracy reaches upto desired level

% looping through all subjects and cropping faces if found
% extract the subject photo and crop faces and saving it in to respective
% folders
for i =1:n
    str = ['s0',int2str(i)];
    ds1 = imageDatastore(['photos\',str],'IncludeSubfolders',true,'LabelSource','foldernames');
    cropandsave(ds1,str);
end
 im = imageDatastore('croppedfaces','IncludeSubfolders',true,'LabelSource','foldernames');
 % Resize the images to the input size of the net
 im.ReadFcn = @(loc)imresize(imread(loc),[227,227]);
 [Train ,Test] = splitEachLabel(im,0.8,'randomized');

 %  Test Layer
varSize = 227;
conv1 = convolution2dLayer(5,varSize,'Padding',2,'BiasLearnRateFactor',2);
% conv1.Weights = gpuArray(single(randn([5 5 3 varSize])*0.0001));
fc1 = fullyConnectedLayer(64,'BiasLearnRateFactor',2,'WeightLearnRateFactor',10);
% fc1.Weights = gpuArray(single(randn([64 576])*0.1));
fc2 = fullyConnectedLayer(4,'BiasLearnRateFactor',2,'WeightLearnRateFactor',10);
% fc2.Weights = gpuArray(single(randn([4 64])*0.1));

%RGB Image so the varsize is 3
layers = [
    imageInputLayer([varSize varSize 3]);
    
    conv1;
    maxPooling2dLayer(3,'Stride',2);
    reluLayer();
    
    convolution2dLayer(5,227,'Padding',2,'BiasLearnRateFactor',2);
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
    'InitialLearnRate', 0.0001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 7, ...
    'MiniBatchSize', 10, ...
    'Plots','training-progress', ...
    'Verbose', true);
% Test later end

 [newnet,info] = trainNetwork(Train, layers, opts);
 [predict,scores] = classify(newnet,Test);
 names = Test.Labels;
 pred = (predict==names);
 s = size(pred);
 acc = sum(pred)/s(1);
 fprintf('The accuracy of the test set is %f %% \n',acc*100);
% Test a new Image
% use code below with giving path to your new image
 img = imread('test_photo\img_1.jpg');
 [img,face] = cropface(img);
 % face value is 1 when it detects face in image or 0
 if face == 1
   img = imresize(img,[227 227]);
   predict = classify(newnet,img);
 end
 nameofs01 = 'Dylan';
 nameofs02 = 'Fatin';
 nameofs03 = 'Yvonne';
 nameofs04 = 'Syafiq';
 if predict=='s01'
   fprintf('The face detected is %s',nameofs01);
 elseif  predict=='s02'   
     fprintf('The face detected is %s',nameofs02);
 elseif  predict=='s03'
   fprintf('The face detected is %s',nameofs03);
 elseif  predict=='s04'
     fprintf('The face detected is %s',nameofs04);
 end	 
 [predict,score] = classify(newnet,img)
 fprintf('predict %s\n',predict);
 fprintf('score is %d\n',score);