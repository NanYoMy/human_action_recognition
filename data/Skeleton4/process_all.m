%process all data
%read_skeleton_file('D:\я╦ювобть\nturgbd_skeletons\nturgb+d_skeletons\S001C001P003R001A060.skeleton')
Dir='D:\я╦ювобть\nturgbd_skeletons\nturgb+d_skeletons\';
% Dir='.\skeleton\'
DirOut='D:\я╦ювобть\nturgbd_skeletons\mat\'
fileFolder=fullfile(Dir);
 
dirOutput=dir(fullfile(fileFolder,'*.skeleton'));
 
fileNames={dirOutput.name};


for i =1:size(fileNames,2)
    
    abs_path = strcat(Dir, char(fileNames(i)));
    [count,rgb]=read_skeleton_file(abs_path);
    
%     if size(rgb,1)==2
%         crop(rgb);
%         toRGB(abs_path,squeeze(rgb(2,:,:,:)));
%     end
%     toRGB(abs_path,squeeze(rgb(1,:,:,:)));
    output=strcat(DirOut, char(fileNames(i)),'.mat');
    save(output,'rgb')
end
fprint('finish')