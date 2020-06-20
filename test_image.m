[predict,scores] = classify(newnet,Test);
 names = Test.Labels;
 pred = (predict==names);
 s = size(pred);
 acc = sum(pred)/s(1);
 fprintf('The accuracy of the test set is %f %% \n',acc*100);
% Test a new Image
% use code below with giving path to your new image
 img = imread('test_photo\img_3.jpg');
 imgs = img;
 [img,face] = cropface(img);
 % face value is 1 when it detects face in image or 0
 if face == 1
   img = imresize(img,[227 227]);
   predict = classify(newnet,img)
 end
 nameofs01 = 'name of subject 1';
 nameofs02 = 'name of subject 2';
 nameofs03 = 'name of subject 3';
 if predict=='s01'
   fprintf('The face detected is %s',nameofs01);
 elseif  predict=='s02'%   fprintf('The face detected is %s',nameofs02);
 elseif  predict=='s03'
   fprintf('The face detected is %s',nameofs03);
 end	 
 [predict,score] = classify(newnet,img)
%  fprintf('predict %s\n',predict);

imshow(imgs);
 if predict=='s01'
   fprintf('The face detected is Dylan \n');
   title('Dylan');
 elseif  predict=='s02'%  
     fprintf('The face detected is Fatin \n');
   title('Fatin');
 elseif  predict=='s03'
   fprintf('The face detected is Yvonne \n');
   title('Yvonne');
 elseif  predict=='s04'
   fprintf('The face detected is Syafiq \n');
   title('Syafiq');
 end
fprintf('score is %d\n',score);