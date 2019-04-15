function [ output_args ] = toRGB(path,img )
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
    refindex=1;
    ref=img(refindex,:,:);
    img=img(:,:,:)-repmat(ref,size(img,1),1);
    img(refindex,:,:)=[];
    
    

    rmn=min(min(img(:,:,1)));
    rmx=max(max(img(:,:,1)));
    gmn=min(min(img(:,:,2)));
    gmx=max(max(img(:,:,2)));  
    bmn=min(min(img(:,:,3)));
    bmx=max(max(img(:,:,3)));
    
    normalfacotr=[0,0,0];
    normalfacotr(1)=gmx-gmn;
    normalfacotr(2)=bmx-bmn;
    normalfacotr(3)=rmx-rmn;  
    
    
    img(:,:,1)=(img(:,:,1)-rmn)/(max(normalfacotr));
    imshow(img(:,:,1))
    r_file=sprintf('%s_r.png',path);
    imwrite(img(:,:,1),r_file)
    
    img(:,:,2)=(img(:,:,2)-gmn)/(max(normalfacotr));
    imshow(img(:,:,2))
    g_file=sprintf('%s_g.png',path);
    imwrite(img(:,:,2),g_file)
    

    img(:,:,3)=(img(:,:,3)-bmn)/(max(normalfacotr));
    imshow(img(:,:,3))
    g_file=sprintf('%s_b.png',path);
    imwrite(img(:,:,3),g_file)

    imshow(img);

    imwrite(img,sprintf('%s.png',path))
    
end

