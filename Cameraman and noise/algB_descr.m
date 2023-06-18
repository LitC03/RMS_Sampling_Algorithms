function [M,seen,path_taken]=algB_descr(I, sp, sigma,total_runs)
% Algorithm B for sparse sampling of Hyperspectral images.

% INPUTS: 
% I: mean across raman spectra
% sp: scattering probability
% sigma: standard deviation of Gaussian filter
% total_runs: amount of pixels that will be sampled

% OUTPUTS:
% M: matrix with the same dimensions as data with the values that have been
% sampled in their respective position
% seen: logical matrix with the positions that have been sampled as true
% (and false if they haven't been sampled)
% path taken: matrix with the same dimensions as data, with zeros
% everywhere except for the positions that have been sampled. In these, the
% order of the sampling is indicated (example: the first pixel that is
% sampled is given the number 1, the second will be given the number 2...)

    [M,runs_left,seen, path_taken,nan_matr] = sample_uniform(I,total_runs);
    [M,seen, path_taken]=sample_interp(M,seen,path_taken,runs_left,total_runs,sp,sigma,nan_matr,I);
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [M,seen,nan_matr,path_taken]=sample_corners(I,dims,M,seen,nan_matr)
    path_taken = zeros(dims);
    pos = [1,1; 1, dims(2);dims(1), 1;dims(1), dims(2)];
    for a=1:size(pos,1)
        M(pos(a,1),pos(a,2),:)= I(pos(a,1),pos(a,2),:);
        seen(pos(a,1),pos(a,2))=1;
        nan_matr(pos(a,1),pos(a,2))=NaN;
        path_taken(pos(a,1),pos(a,2))=a;
    end

end
function [paper, filter] = create_filter(dims)
    %% Create filter for high frequencies
    r=round(max(dims)*0.9);
    x_c=round(dims(2)/2);
    y_c=round(dims(1)/2);
    % r=round(max([dimensions(1),dimensions(2)])*0.5);x_c=round(dimensions(2)/2);y_c=round(dimensions(1)/2);
    %generate a coordinate grid
    [y,x]=ndgrid(1:dims(1),1:dims(2));
    %perform calculation
    paper= (x-x_c).^2+(y-y_c).^2 <= r^2;
    % paper=~paper;
    
    r=round(max(dims)*0.5);x_c=round(dims(2)/2);y_c=round(dims(1)/2);
    %generate a coordinate grid
    [y,x]=ndgrid(1:dims(1),1:dims(2));
    %perform calculation
    circle2= (x-x_c).^2+(y-y_c).^2 <= r^2;
    circle2=~circle2;
    
    paper=paper.*circle2;
    filter = log(abs(ifft2(fftshift(paper))));
end

function [M,seen, path_taken]=sample_interp(M,seen,path_taken,runs_left,total_runs,th,sigma,nan_matr,I)
    dims = size(I);
    for a=1:runs_left
        ranNum = rand;
        if ranNum <= th
            %Get the pixel which has the most probability of being information-rich
            [~,indx] = max(bwdist(double(seen)).*nan_matr,[],'all','linear');
            [row,col] = ind2sub(dims,indx); %get 2d coordinates
        
            M(row,col)=I(row,col); % save value
           
            seen(row,col)=true; % note that we have sampled this location
            nan_matr(row,col) = NaN; % make sure we will not sample it later
        else
            [M_new,seen_new,nan_matr_new,row,col]=diff_int(M,seen,nan_matr,I,dims,sigma);
            M=M_new;
            seen = seen_new;
            nan_matr = nan_matr_new;
        end
        path_taken(row,col)=total_runs-runs_left+a;
    end

end

function [M,runs_left,seen, path_taken,nan_matr] = sample_uniform(I,total_runs)
    dims= size(I);
    M = zeros(dims); %this is the matrix with the information I will be sampling
    seen =false(dims); %This matrix has a value of 1 in the places that have already been sampled
    nan_matr =ones(dims); %This matrix has a value of 1 in the places that have already been sampled
    [xq,yq] = meshgrid(1:dims(2),1:dims(1));
    
    stepx=dims(1)-1;
    stepy=dims(2)-1;
    
    energy=0;
    rounds=1;
    increased=0;
    decreased=0;

    [paper, ~] = create_filter(dims);

    [M,seen,~,path_taken]=sample_corners(I,dims,M,seen,nan_matr);

    runs_left = total_runs-4;
    n=1;
    while (n+4<=total_runs) && (~increased)
        for r=1:stepx:dims(1)
            for c=1:stepy:dims(2)
                if seen(r,c)==false
                    M(r,c) = I(r,c);
                    runs_left=runs_left-1; % the next part of the algorithm will run one less time
                    seen(r,c) = true; % Note that you have sampled that position
                    nan_matr(r,c) = NaN;
                    path_taken(r,c) = n+4;
                    n=n+1;
                    if n>total_runs
                        break;
                    end
                end
                if n>total_runs
                   break;
                end
            end
            if n>total_runs
                break;
            end
        end
        %% Interpolation
        [x,y]=find(double(seen));
    
        ind = sub2ind(dims,x,y); 
        SqueezedImage = M(:);
        Inter_matrix = SqueezedImage(ind);
    
        Cubic_result = griddata(x,y,Inter_matrix,yq,xq,'cubic');
        Cubic_result(isnan(Cubic_result))=0;
        curr_est=Cubic_result;

        F = fftshift(fft2(curr_est)); 
        old_en = energy;

        energy = (1/2*pi)*sum(abs(F.*conj(F).*paper).^2,'all'); %energy calculation
         if (old_en>energy)
            decreased=1;
         end
        if (energy>old_en)&&(decreased)
           increased=1;
        end

        stepx=floor(stepx/2);
        stepy=floor(stepy/2);

        rounds=rounds+1;
    end
end

function [M,seen,nan_matr,row,col]=diff_int(M,seen,nan_matr,I,dims,sigma)
    x_space = repelem(1:dims(1),dims(2));
    y_space = repmat(1:dims(2),1,dims(1));
    [xq,yq] = meshgrid(1:dims(2),1:dims(1));
    [x,y]=find(seen);
    
    indxs = sub2ind(dims,x,y);
    inter_mat  = M(indxs);
    
    NN_interpolation =scatteredInterpolant(x,y,inter_mat,'nearest');
    NN_result = NN_interpolation(y_space,x_space);
    NN_matr_result=reshape(NN_result,dims);
    Cubic_result = griddata(x,y,inter_mat,yq,xq,'cubic');
    Cubic_result(isnan(Cubic_result))=0;
    
    diff=abs(NN_matr_result-Cubic_result);

    edges=edge(Cubic_result);
    
    diff=diff.*imgaussfilt(double(edges),sigma);

    diff=diff.*bwdist(double(seen));

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
