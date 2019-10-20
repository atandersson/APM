function [x, y] = generateData(a,b,c,n,d)
    x    = zeros(d,n); % to store x
    y    = zeros(1,n); % to store y
    for i = 1:n
        x(:,i) = rand(d,1)*6;
        y(i)   = respFunc(a,b,c,x(:,i));
    end
end