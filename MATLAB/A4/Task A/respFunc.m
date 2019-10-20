function y = respFunc(a,b,c,x)
    exponent = 0;
    for i=1:length(b)
        exponent = exponent-b(i)*(x(i)-c(i))^2;
    end
    y = a*exp(exponent);
end