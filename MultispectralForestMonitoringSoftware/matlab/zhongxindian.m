 [l,num]= bwlabel(tu,8);
plot_x=zeros(1,num);%%用于记录质心位置的坐标
plot_y=zeros(1,num);
[height,width]=size(l);
for k=1:num  %%num个区域依次统计质心位置
    sum_x=0;sum_y=0;area=0;
    for i=1:height
    for j=1:width
       if l(i,j)==k
        sum_x=sum_x+i;
        sum_y=sum_y+j;
        area=area+1;   
       end
    end
    end
    if area>50
    plot_x(k)=fix(sum_x/area);
    plot_y(k)=fix(sum_y/area);
    else
        plot_x(k)=nan;
        plot_y(k)=nan;
    end
end