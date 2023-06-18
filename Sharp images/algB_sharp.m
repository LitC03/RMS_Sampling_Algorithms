% AGLORITHM B: uses energy approach to sample an hyperspectral image and
% gives the RMSE and compares it to uniform sampling

clear
close all
rng('default');
set(groot,'defaultLegendInterpreter','latex','defaultAxesTickLabelInterpreter','latex' ...
    , 'defaultAxesFontSize',12,'DefaultTextInterpreter','latex')

%% Get data
% data = zeros(50);
% data(19:31,21:29)=1;
% data(20:30,22:28)=0;
% % I(10:40,25)=1;
% name = 'rect_edges';

% data = zeros(50);
% data(10:40,25)=1;
% data(25,10:40)=1;
% name = 'cross';

% data = zeros(50);
% data(2:10,2:10)=1;
% data(30:48,30:48)=1;
% name = 'rect2';

cdata=imread('diagonal_lines.gif');
data=double(cdata(2:51,2:51));
name='zigzag';

%% Set parameters
r_factor=4;
c_factor=3;
sigma = 0.9;
sp = 0.5;

%% Display original image
f=figure(1);
f.Position = [551,447.6666666666666,257.3333333333333,184.6666666666666];
imagesc(data)
colormap(turbo)
% colormap(gray)

N = numel(data); % get the total n0 of pixels
dims = size(data); %get the dimensions of the image

%% Compare with equally - spaced sampling
eq_samp = NaN(dims);
eq_samp(1:r_factor:end,1:c_factor:end) = data(1:r_factor:end,1:c_factor:end);
total_runs = nnz(~isnan(eq_samp));

%% Interpolation equally-spaced sampling
        
%%%%%%%%%%%%%%%%%%%%%%%% NN, Cubic %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[NN_eq] = NN_interp(eq_samp,dims,~isnan(eq_samp));
% [Cubic_eq] = Cubic_interp(eq_samp,dims,~isnan(eq_samp));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% eq_samp_thinplate = interpolate_thin_plate(eq_samp,~isnan(eq_samp),dims);

f=figure(6);
f.Position = [287,445.6666666666666,250,191.3333333333334];
% imagesc(eq_samp_thinplate)
clim([min(data,[],'all') max(data,[],'all') ]);
imagesc(NN_eq)
% colormap(turbo)
colormap(gray)

 %% Search information-rich locations

[M,seen,path_taken]=algB_descr(data, sp, sigma,total_runs);

%% Interpolation - thin plate spline
        
%%%%%%%%%%%%%%%%%%%%%%%%%% NN, Cubic %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[NN_M] = NN_interp(M,dims,seen);
% [Cubic_M] = Cubic_interp(M,dims,seen);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% M_thinplate = interpolate_thin_plate(M,seen,dims);

f=figure(5);
f.Position = [18.333333333333332,443,264,194];
% imagesc(M_thinplate)
clim([min(data,[],'all') max(data,[],'all') ]);
imagesc(NN_M)
% colormap(turbo)
colormap(gray)

%% Error calculation
% E1 = sqrt(sum((data-eq_samp_thinplate).^2,'all')./N)
% E2 = sqrt(sum((data-M_thinplate).^2,'all')./N)
E1 = sqrt(sum((data-NN_eq).^2,'all')./N)
E2 = sqrt(sum((data-NN_M).^2,'all')./N)
if E2 < E1
    disp('Success!')
else
    disp('Unsuccesful search')
end

%% Save
% save(['algB_',name,'_F',num2str(r_factor),'X',num2str(c_factor),'_std_',num2str(sigma),'_sp',num2str(sp),'.mat'],'M','seen','data','E1','E2','eq_samp','total_runs','M_thinplate','eq_samp_thinplate','r_factor','c_factor','path_taken','sp','sigma')
save(['algB_',name,'_F',num2str(r_factor),'X',num2str(c_factor),'_std_',num2str(sigma),'_sp',num2str(sp),'.mat'],'M','seen','data','E1','E2','eq_samp','total_runs','NN_M','NN_eq','r_factor','c_factor','path_taken','sp','sigma')



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [NN_mat] = NN_interp(M,dims,seen)

    x_space = repelem(1:dims(1),dims(2));
    y_space = repmat(1:dims(2),1,dims(1));
    [x,y]=find(seen);
    
    % M(isnan(eq_samp))=0;
    
    indxs = sub2ind(dims,x,y);
    inter_mat  = M(indxs);
    
    NN_interpolation =scatteredInterpolant(x,y,inter_mat,'nearest');
    NN_result = NN_interpolation(y_space,x_space);
    NN_mat=reshape(NN_result,dims);
end

function [a, xc]=train_thin_plate_spline(x,y)
    xc=x;%basis function centres
    D = pdist2(xc.', x.').';
    
    % Convert to symmetric matrix
    D = D + D.';
    D = D - diag(diag(D));
    D=D/2;
    E=D.^2.*log(D);%apply thin plate spline RBF
    E(D==0)=0;%avoid -inf
    
    a=y/E;%use mrdivide to solve system of equations. For large systems it may 
    %make more sense to use an online method such as recursive least squares
    %(RLS)
end

function y_hat=sim_thin_plate_spline(x,x_c,a)

    N_p=size(x,2);%number of points
    
    X = repmat(x_c,[1,1,N_p]);
    P_new=sum((permute(X,[1,3,2])-x).^2);
    sP_new=sqrt(squeeze(P_new));
    E2=sP_new.^2.*log(sP_new);
    E2(sP_new==0)=0;
    y_hat_new=a*E2';
    y_hat=y_hat_new;

end

function M_thinplate = interpolate_thin_plate(M,seen,dims)
    SqueezedImage= M(:)';

    [x_t(1,:),x_t(2,:)]=find(seen);%training input
    ind = sub2ind(dims,x_t(1,:),x_t(2,:)); 
    Y_t = SqueezedImage(ind);
    
    [a, Xc]=train_thin_plate_spline(x_t,Y_t);%train thin plate spline network
    
    X_range=[1 dims(2);1 dims(1)];%limits of X to plot at
    N_grid=[dims(1) dims(2)];%number of points to plot at in form [x y]
    [x_mesh,y_mesh]=meshgrid(linspace(X_range(1,1),X_range(1,2),N_grid(2)),linspace(X_range(2,1),X_range(2,2),N_grid(1)));
    X_mesh=[reshape(y_mesh,1,[]);reshape(x_mesh,1,[])];%reshape into X=[x;y] matrix
    
    Y_mesh=sim_thin_plate_spline(X_mesh,Xc,a);%perform tps calculation on mesh points
    M_thinplate=reshape(Y_mesh,N_grid(1),N_grid(2));%reshape into rectangular matrix

end

function [Cubic_result] = Cubic_interp(M,dims,seen)
    [xq,yq] = meshgrid(1:dims(2),1:dims(1));
    [x,y]=find(seen);
    
    indxs = sub2ind(dims,x,y);
    inter_mat  = M(indxs);

    Cubic_result = griddata(x,y,inter_mat,yq,xq,'cubic');
    Cubic_result(isnan(Cubic_result))=0;

end