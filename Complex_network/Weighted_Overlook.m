function adjmatrix=Weighted_Overlook(time_series)
[w,m,~]=size(time_series);
adjmatrix = zeros(m,m);
min_value=min(time_series);
if (min_value<0)
    time_series = time_series+abs(min_value);
end
for i = 1:m
    for j = i+1:m 
        if(time_series(i)>time_series(j))
            adjmatrix(i,j) = (time_series(i)-time_series(j))/abs(j-i); adjmatrix(j,i) = -1*(time_series(i)-time_series(j))/abs(j-i);
        elseif(time_series(i)==time_series(j))
            adjmatrix(i,j) = 0; adjmatrix(j,i) = 0;
        else
            adjmatrix(i,j) = -1*(time_series(j)-time_series(i))/abs(j-i); adjmatrix(j,i) = (time_series(j)-time_series(i))/abs(j-i);
        end
    end
end
