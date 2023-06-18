function [M,seen,prob_map,path_taken]=algA_descr(data, sp, sigma,total_runs)
% Algorithm A for sparse sampling of Hyperspectral images.

% INPUTS: 
% data: mean across raman spectra
% sp: scattering probability
% sigma: standard deviation of Gaussian filter
% total_runs: amount of pixels that will be sampled

% OUTPUTS:
% M: matrix with the same dimensions as data with the values that have been
% sampled in their respective position
% seen: logical matrix with the positions that have been sampled as true
% (and false if they haven't been sampled)
% prob_map: matrix with the same dimensions as data that is the result of
% the convolution with the Gaussian filter
% path taken: matrix with the same dimensions as data, with zeros
% everywhere except for the positions that have been sampled. In these, the
% order of the sampling is indicated (example: the first pixel that is
% sampled is given the number 1, the second will be given the number 2...)

    dims = size(data);
    
    M=zeros(dims); % Matrix that will be filled with values of sampled locations
    w=zeros(dims); % Weights for convolution
    prob_map=ones(dims)./numel(data);
    seen = false(dims); % Logical matrix with ones in locations that have been sampled
    nan_matr = ones(dims); % used to ignore already-sampled pixels during sampling
    
    [M,seen,nan_matr,path_taken]=sample_corners(data,dims,M,seen,nan_matr);
    % Sample 4 corners

    for a=1:total_runs-4
        r_num = rand; % Select a random number between 1 and 0
        if r_num<sp
        %Get the pixel which has the most probability of being information-rich
        search_grid=prob_map.*nan_matr; % multiply with a matrix that has nan 
        % values in the locations that have already been sampled
        [max_val,~] = max(search_grid,[],'all','linear'); % search for the biggest value
        log_m = search_grid==max_val; % find the places where that value is located
        [row,col] = find(log_m); % obtain their indeces
        ind_length = round(length(row)/2); % it there are more than one possible places, choose the middle one
        row=row(ind_length);
        col=col(ind_length);
    
        M(row,col)=data(row,col); % save value
       
        seen(row,col)=true; % note that we have sampled this location
        nan_matr(row,col) = NaN; % make sure we will not sample it later
    
        av = mean(M(seen),"all"); % calculate average value of the image
        w(seen) = M(seen)-av; % Calculate the deviation of each pixel from the average
    
        modulation = imgaussfilt(w,sigma); %Convolve gaussian with a specific std deviation
        prob_map = modulation;
        
        else
            [M,seen,nan_matr,row,col]=diff_interpolations(M,seen,nan_matr,data,dims);
        end
        path_taken(row,col)=a+4; % save the places that have been visited before
        
        figure(3)
        imagesc(M)
        title('Estimate')

        figure(4)
        imagesc(seen)
        title('Sampling points')


        figure(2)
        imagesc(prob_map)
        title('Prob map')
        
    end



end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [M,seen,nan_matr,row,col]=diff_interpolations(M,seen,nan_matr,I,dims)

x_space = repelem(1:dims(1),dims(2));
y_space = repmat(1:dims(2),1,dims(1));
[xq,yq] = meshgrid(1:dims(2),1:dims(1));
[x,y]=find(seen); % find the places that have been sampled already

indxs = sub2ind(dims,x,y);
inter_mat  = M(indxs); %get the values than have been sampled already in an array

NN_interpolation =scatteredInterpolant(x,y,inter_mat,'nearest'); 
NN_result = NN_interpolation(y_space,x_space); % get the NN interpolation
NN_matr_result=reshape(NN_result,dims);
Cubic_result = griddata(x,y,inter_mat,yq,xq,'cubic'); % get the bicubic interpolation
Cubic_result(isnan(Cubic_result))=0;

diff=abs(NN_matr_result-Cubic_result); % find the place where the difference between interpolations is maximum
diff=diff.*bwdist(double(seen)); % Multiply with distance transform of sampled locations

search_grid=diff.*nan_matr; 
[max_val,~] = max(search_grid,[],'all','linear');
log_m = search_grid==max_val;
[row,col] = find(log_m);
ind_length = round(length(row)/2);
row=row(ind_length);
col=col(ind_length);
M(row,col)=I(row,col);
seen(row,col)=true;
nan_matr(row,col)=NaN;
end

function [M,seen,nan_matr,path_taken]=sample_corners(I,dims,M,seen,nan_matr)
    path_taken = zeros(dims);
    pos = [1,          1; 
           1,       dims(2);
           dims(1),    1;
           dims(1), dims(2)];
    for a=1:size(pos,1)
        M(pos(a,1),pos(a,2),:)= I(pos(a,1),pos(a,2),:);
        seen(pos(a,1),pos(a,2))=1;
        nan_matr(pos(a,1),pos(a,2))=NaN;
        path_taken(pos(a,1),pos(a,2))=a;
    end

end