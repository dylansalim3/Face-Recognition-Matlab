function output_value = load_database()

persistent loaded;
persistent numeric_Image;

for i=1:5
    cd(strcat('s',num2str(i)));
    for j=1:4
        image_Container = imread(strcat(num2str(j),'.jpg'));
        all_Image = imresize(image_Container,[256,256]);
        all_Image= rgb2gray(all_Image);
%        imshow(all_Image);
       all_Images(:,(i-1)*4+j)=reshape(all_Image,size(all_Image,1)*size(all_Image,2),1);
    end
        display('Doading Database');
        cd ..
    end
    numeric_Image = uint8(all_Images);
loaded = 1;
output_value = numeric_Image;