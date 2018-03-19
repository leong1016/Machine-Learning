function w = gradientDescent(x,y)
    w = [0 0 0 0];
    t = 0;
    while t < 1000
        w = w - 0.05 * gradient(w,x,y);
        t = t + 1;
    end
end

