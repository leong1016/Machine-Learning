function res = EntropyDouble(a)
    res = - (a*log2(a)+(1-a)*log2(1-a));
end