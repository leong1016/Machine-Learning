function result = gradient(w,x,y)
    result = zeros(1,length(w));
    for j=1:length(w)
        sum = 0;
        for i=1:length(x)
            disp(x(i,:))
            sum = sum + (y(i)-dot(w,x(i,:)))*x(i,j);
        end
        result(j) = -sum;
    end
end