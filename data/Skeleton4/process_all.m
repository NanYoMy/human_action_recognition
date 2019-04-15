%process all data
%read_skeleton_file('D:\я╦ювобть\nturgbd_skeletons\nturgb+d_skeletons\S001C001P003R001A060.skeleton')
Dir='D:\я╦ювобть\nturgbd_skeletons\nturgb+d_skeletons\\';
fileFolder=fullfile(Dir);
 
dirOutput=dir(fullfile(fileFolder,'*.skeleton'));
 
fileNames={dirOutput.name};


for i =1:size(fileNames,2)
    
    abs_path = strcat(Dir, char(fileNames(i)));
    bodyinfo=read_skeleton_file(abs_path);
   
end
print('finish')