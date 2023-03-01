clear 
clc
i=imread('test.png');
i(1:10,1:10,1)
qqq(:,:,1)=i(:,:,2);
qqq(:,:,2)=i(:,:,3);
qqq(:,:,3)=i(:,:,3);
www=i;
i=double(i);
b1(:,:)=i(:,:,2);
b2(:,:)=i(:,:,3);
b11(:,:)=i(:,:,1);

%NDVI计算
 b=(b1-b2)./(b1+b2);
 [m,n]=size(b);
tu=b*0;
for q=1:m
 for w=1:n

%  if b(q,w)<0&&b(q,w)>-1000&&www(q,w,3)>25
  if b(q,w)<0.2&&qqq(q,w,1)>0.025&&b(q,w)>0&&qqq(q,w,2)>0.025
tu(q,w)=1;

 end


 end


end
figure(1)
imshow(tu);
B=ones(1);
% tu=imdilate(tu,B);

% se1=strel('disk',1);
% tu=imerode(tu,se1);
% se1=strel('disk',1);
% tu=imerode(tu,se1);
% se1=strel('disk',1);
% tu=imerode(tu,se1);
% B=ones(3);
%
% tu=imdilate(tu,B);

%膨胀腐蚀操作
B=ones(5);

tu=imdilate(tu,B);
se1=strel('disk',5);
tu=imerode(tu,se1);
B=ones(15);

tu=imdilate(tu,B);
se1=strel('disk',14);
tu=imerode(tu,se1);



se1=strel('disk',1);
tu=imerode(tu,se1);

figure(2)
imshow(tu);

figure(3)
imshow(b2*5);
% B=ones(30);
%
% tu=imdilate(tu,B);
% % %  [l,num]= bwlabel(tu,8);
% % % plot_x=zeros(1,num);%%用于记录质心位置的坐标
% % % plot_y=zeros(1,num);
% % % [height,width]=size(l);
% % % for k=1:num  %%num个区域依次统计质心位置
% % %     sum_x=0;sum_y=0;area=0;
% % %     for i=1:height
% % %     for j=1:width
% % %        if l(i,j)==k
% % %         sum_x=sum_x+i;
% % %         sum_y=sum_y+j;
% % %         area=area+1;   
% % %        end
% % %     end
% % %     end
% % %     if area>50
% % %     plot_x(k)=fix(sum_x/area);
% % %     plot_y(k)=fix(sum_y/area);
% % %     else
% % %         plot_x(k)=nan;
% % %         plot_y(k)=nan;
% % %     end
% % % end


% imwrite(tu,'1.tif','Compression','none');