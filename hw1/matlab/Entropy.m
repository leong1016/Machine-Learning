function res = Entropy(a1, a2, b1, b2)
    a = a1 / (a1+a2);
    b = b1 / (b1+b2);
    s = a1 + a2 + b1 + b2;
    if (a==0 || a==1)
        ha = 0;
    else
        ha = - (a*log2(a)+(1-a)*log2(1-a));
    end
    if (b==0 || b==1)
        hb = 0;
    else
        hb = - (b*log2(b)+(1-b)*log2(1-b));
    end
    res = (a1+a2)/s * ha + (b1+b2)/s * hb
end