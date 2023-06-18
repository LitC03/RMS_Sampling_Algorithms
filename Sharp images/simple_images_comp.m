% Compare simple images
close all
%% Compare everything
load("algA_zigzag_F4X3_std_0.5_sp0.6.mat")
f=figure(7);
f.Position = [129,305.6666666666666,798,192.6666666666666];
subplot(1,4,1)
imagesc(data)
title('\bf{Original Image}')
set(gca,'YTickLabel',[],'XTickLabel',[]);
subplot(1,4,2)
imagesc(NN_eq)
title('\bf{Equally-spaced sampling}')
subtitle(['RMSE: ',num2str(E1)])
set(gca,'YTickLabel',[],'XTickLabel',[]);
subplot(1,4,3)
imagesc(NN_M)
title('\bf{Algorithm A}')
subtitle(['RMSE: ',num2str(E2)])
set(gca,'YTickLabel',[],'XTickLabel',[]);
subplot(1,4,4)

f=figure(2);
f.Position = [119.6666666666667,218.3333333333333,978.6666666666666,278];

[x,y] = find(~isnan(eq_samp));
subplot(1,3,1)
imagesc(NN_eq)
clim([min(data,[],'all') max(data,[],'all') ]);
colormap(turbo)
hold on
scatter(y,x,'r','filled','SizeData',9);
set(gca,'YTickLabel',[],'XTickLabel',[]);
title('\bf{Equally-spaced sampling}')

[x,y] = find(seen);
subplot(1,3,2)
imagesc(NN_M)
clim([min(data,[],'all') max(data,[],'all') ]);
hold on
scatter(y,x,'r','filled','SizeData',9);
set(gca,'YTickLabel',[],'XTickLabel',[]);
title('\bf{Algorithm A}')
colormap(gray)

load('algB_zigzag_F3X4_std_0.9_sp0.5.mat')
figure(7)
subplot(1,4,4)
imagesc(NN_M)
title('\bf{Algorithm B}')
subtitle(['RMSE: ',num2str(E2)])
set(gca,'YTickLabel',[],'XTickLabel',[]);

colormap(gray)

figure(2)
[x,y] = find(seen);
subplot(1,3,3)
imagesc(NN_M)
clim([min(data,[],'all') max(data,[],'all') ]);
hold on
scatter(y,x,'r','filled','SizeData',9);
set(gca,'YTickLabel',[],'XTickLabel',[]);
title('\bf{Algorithm B}')