%
% plots a function + std
%

x = [0:0.01:1];
% bogus data, function + std, just for illustration!!
% you should use your own data, functions + std!!!
Xn = [0.3;  ;0.5; 0.9];
Tn = [0.05; -0.35; 0.8];
mx = x.^2;
sx = 0.2*sqrt(x);


%
%
%
figure(111)
axis([0 1 -1.5 1.5]);
hold on
% first the m(x)+/-s(x) areas (no line)
area(x,(mx+sx), 'FaceColor', [1.0, 0.8, 0.8], 'BaseValue',-1.5);  % pinkish
area(x,(mx-sx), 'FaceColor', [1.0, 1.0, 1.0], 'BaseValue',-1.5);  % white
% the lines for the predictive mean m(x) and variance s(x) around it
plot(x,(mx+sx),'r', 'LineWidth',2);     % red
plot(x,mx,'k');                         % black
plot(x,(mx-sx),'r', 'LineWidth',2);     % red
% circle the datapoints
plot(Xn,Tn,'o','MarkerEdgeColor','k','LineWidth',2, 'MarkerSize',10);

%
%
disp('press any key');
pause;
% 
% five made-up functions, again only for illustration for the second matlab plot!!!
% you should generate your own functions !!
for i=1:5,
    %a = randn(2,1);
    y = cos(2*pi*i*x)/5+(i-3)/2; % these are not the function that you need!
    plot(x,y,'b','LineWidth',1.5); 
end;
%
%
disp('press key to close figure');
pause;
% 
close(111);