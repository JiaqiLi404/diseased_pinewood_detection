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
