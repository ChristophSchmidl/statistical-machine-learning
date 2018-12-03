% some ideas to make a scatter plot in the logistic regression exercise
% you may use and adapt this code according to your needs 

% just a random set as an example
x = randn(1000,2);
% "class probabilities"
cl = 1./(1+exp(-x(:,1))); 

% this seems to work
mycolormap = colormap('Jet');
d64 = [0:63]/63; % 
c = interp1(d64, mycolormap,cl);
dotsize = 10;
scatter(x(:,1),x(:,2),dotsize,c,'fill');
xlabel('x_1');
ylabel('x_2');
title('a nice scatterplot');
colorbar; % what do the colors mean?