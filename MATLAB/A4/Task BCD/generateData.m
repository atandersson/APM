function [x, y] = generateData(a,b,c,n,d)
    x    = zeros(n,d); % to store x
    y    = zeros(n,1); % to store y
    for i = 1:n
        x(i,:) = rand(1,d)*6;
        y(i)   = respFunc(a,b,c,x(i,:));
    end
end