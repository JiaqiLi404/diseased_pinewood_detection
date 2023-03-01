info = geotiffinfo('11111.tif');
[x,y] = pix2map(info.RefMatrix, 1, 1);%最后两个是行列数
[lat,lon] = projinv(info, x,y)