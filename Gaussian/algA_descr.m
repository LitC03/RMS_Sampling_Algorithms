function [M,seen,prob_map,path_taken]=algA_descr(data, sp, sigma,total_runs)
% Algorithm A for sparse sampling of Hyperspectral images.

% INPUTS: 
% data: mean across raman spectra (if the image is non-hyperspectral, this
% variable will be equal to the image)
% sp: scattering probability
% sigma: standard deviation of Gaussian filter
% total_runs: amount of pixels that will be sampled

% OUTPUTS:
% M: matrix with the same dimensions as "data" with the values that have been
% sampled in their respective position
% seen: logical matrix with the positions that have been sampled as true
% (and false if they haven't been sampled)
% prob_map: matrix with the same dimensions as "data" that is the result of
% the convolution with the Gaussian filter
% path taken: matrix with the same dimensions as "data", with zeros
% everywhere except for the positions that have been sampled. In these, the
% order of the sampling is indicated (example: the first pixel that is
% sampled is given the number 1, the second will be given the number 2...)

    dims = size(data);
    
    M=zeros(dims); % Matrix that will be filled with values of sampled locations
    w=zeros(dims); % Matrix that will be convolved with the Gaussian
    prob_map=ones(dims)./numel(data); %initialising this matrix in case sp=0, 
    % (so that there are no errors in the execution)
    seen = false(dims); % Logical matrix that will contain ones in locations that have been sampled
    nan_matr = ones(dims); % Matrix used to ignore already-sampled pixels during sampling
    
    [M,seen,nan_matr,path_taken]=sample_corners(data,dims,M,seen,nan_matr);
    % Sample 4 corners

    for a=1:total_runs-4
        r_num = rand; % Select a random number between 1 and 0
        if r_num<sp
        

        %Get the pixel which has the most probability of being information-rich
        search_grid=prob_map.*nan_matr; % multiply with a matrix that has NaN 
        % values in the locations that have already been sampled - this is
        % done so that locations which have already been sampled are be
        % ignored for sampling

        [max_val,~] = max(search_grid,[],'all','linear'); % search for the biggest value
        % figure(10)
        % imagesc(search_grid)
        loc_m = search_grid==max_val; % find the places where that value is located
        [row,col] = find(loc_m); % obtain their indeces
        ind_length = round(length(row)/2); % it there are more than one possible places, choose the middle one
        row=row(ind_length);
        col=col(ind_length);
    
        M(row,col)=data(row,col); % sample the image
       
        seen(row,col)=true; % note that we have sampled this location
        nan_matr(row,col) = NaN; % make sure we will not sample it later
        
        av = mean(M(seen),"all"); % calculate average value of the sampled locations
        w(seen) = M(seen)-av; % Calculate the deviation of each pixel from the average
    
        prob_map = imgaussfilt(w,sigma); %Convolve gaussian with a specific std deviation
        else
            [M,seen,nan_matr,row,col]=diff_interpolations(M,seen,nan_matr,data,dims);
        end
        path_taken(row,col)=a+4; % save the order of sampling
        

       
    end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [M,seen,nan_matr,row,col]=diff_interpolations(M,seen,nan_matr,data,dims)
    % Get the coordinates of the query locations for the interpolation
    x_space = repmat(1:dims(1),1,dims(2));
    y_space = repelem(1:dims(2),dims(1));
    [xq,yq] = meshgrid(1:dims(2),1:dims(1));
    
    [x,y]=find(seen); % find the places that have been sampled already
    indxs = sub2ind(dims,x,y);
    inter_mat  = M(indxs); %get the values than have been sampled already in an array
    
    NN_interpolation =scatteredInterpolant(x,y,inter_mat,'nearest'); 
    NN_result = NN_interpolation(x_space,y_space); % get the NN interpolation
    NN_matr_result=reshape(NN_result,dims);
    Cubic_result = griddata(x,y,inter_mat,yq,xq,'cubic'); % get the bicubic interpolation
    Cubic_result(isnan(Cubic_result))=0;
    
    diff=abs(NN_matr_result-Cubic_result); % Get the absolute difference between interpolations
    diff=diff.*bwdist(double(seen)); % Multiply with distance transform of sampled locations
    
    search_grid=diff.*nan_matr; % Make sure already-sampled locations are being ignored
    [max_val,~] = max(search_grid,[],'all','linear');
    loc_m = search_grid==max_val;
    [row,col] = find(loc_m);
    ind_length = round(length(row)/2);
    row=row(ind_length);
    col=col(ind_length);
    
    M(row,col)=data(row,col); % Sample the image at the desired location
    seen(row,col)=true; % Note that sampling has occured
    nan_matr(row,col)=NaN; % Avoid sampling it twice
end

function [M,seen,nan_matr,path_taken]=sample_corners(data,dims,M,seen,nan_matr)
    path_taken = zeros(dims);
    
    % Coordinates of the 4 corners of the image
    pos = [1,          1; 
           1,       dims(2);
           dims(1),    1;
           dims(1), dims(2)];

    for a=1:size(pos,1)
        M(pos(a,1),pos(a,2))= data(pos(a,1),pos(a,2)); % Sample the image
        seen(pos(a,1),pos(a,2))=true; % Note that the location has been sampled
        nan_matr(pos(a,1),pos(a,2))=NaN; % Avoid re-sampling
        path_taken(pos(a,1),pos(a,2))=a; % Save the sampling order
    end

end