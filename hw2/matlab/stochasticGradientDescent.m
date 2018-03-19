function w = stochasticGradientDescent(x,y)
    w = [0 0 0 0];
    t = 0;
    while t < 1
        for i=1:length(y)
            grad = zeros(1,length(w));
            for j=1:length(w)
                grad(j) = (y(i)-dot(w,x(i,:)))*x(i,j);
            end
            disp(grad)
            w = w + 0.1 * grad;
            disp(w)
        end
        t = t + 1;
    end
end