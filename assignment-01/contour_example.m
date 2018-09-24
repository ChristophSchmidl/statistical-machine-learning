% example code to make a contour plot of  a function
%
%

% example function
%
lambda1 = 5;
lambda2 = 3;
omega=3;
a1 = 1;
a2 = 1;

g = @(x,y)(lambda1 / 2) * (x - a1).^2 + (lambda2 / 2) * sin(omega*(y - a2));

% here is the plot
[X,Y] = meshgrid(-1:.2:3, 0:.2:4);
Z = g(X,Y);
contour(X,Y,Z);

hold on;

% some additional data 
xydata = [[0.6 1.3 1.78 1.2 1]', [ 1 3  1 2 1.5 ]'];
plot(xydata(:,1),xydata(:,2),'k-o');
hold off

title('a contour plot and some data');
xlabel('x')
ylabel('y');