function [bodycount_last_frame,rgb] = read_skeleton_file(filename)
% Reads an .skeleton file from "NTU RGB+D 3D Action Recognition Dataset".
% 
% Argrument:
%   filename: full adress and filename of the .skeleton file.
%
% For further information please refer to:
%   NTU RGB+D dataset's webpage: 
%       http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp
%   NTU RGB+D dataset's github page: 
%        https://github.com/shahroudy/NTURGB-D
%   CVPR 2016 paper: 
%       Amir Shahroudy, Jun Liu, Tian-Tsong Ng, and Gang Wang, 
%       "NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis", 
%       in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016
%
% For more details about the provided data, please refer to:
%   https://msdn.microsoft.com/en-us/library/dn799271.aspx
%   https://msdn.microsoft.com/en-us/library/dn782037.aspx

fileid = fopen(filename);
framecount = fscanf(fileid,'%d',1); % no of the recorded frames
rgb=zeros(2,25,framecount,3);
bodycount_last_frame=1;
for f=1:framecount
    
    bodycount = fscanf(fileid,'%d',1); % no of observerd skeletons in current frame
    
    if bodycount==2
        bodycount_last_frame=bodycount;
    end  
   
    for b=1:bodycount
        clear body;
        body.bodyID = fscanf(fileid,'%ld',1); % tracking id of the skeleton
        arrayint = fscanf(fileid,'%d',6); % read 6 integers

        lean = fscanf(fileid,'%f',2);

        body.trackingState = fscanf(fileid,'%d',1);
        
        body.jointCount = fscanf(fileid,'%d',1); % no of joints (25)
        for j=1:body.jointCount
            jointinfo = fscanf(fileid,'%f',11);

            joint=[jointinfo(1),jointinfo(2),jointinfo(3)];

            trackingState = fscanf(fileid,'%d',1);
            
            rgb(b,j,f,:)=joint;
        end
        
    end
end
fclose(fileid);
end