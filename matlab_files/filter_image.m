function filtered_image = filter_image(input_image, filter);
    [r,c] = size(input_image);
    [r_f,c_f] = size(filter);
    r_start_index = 1+(r_f-1)/2;
    c_start_index = 1+(c_f-1)/2;
    r_end_index   = (r_f-1)/2;
    c_end_index   = (c_f-1)/2;
    filtered_image = zeros (r,c);

    for i=r_start_index:r-r_end_index
        for j=c_start_index:c-c_end_index
           dummy_mtrx = input_image(i-((r_f-1)/2):i+((r_f-1)/2),j-((c_f-1)/2):j+((c_f-1)/2));
           filtered_block = double(dummy_mtrx) .* filter;
           filtered_image(i, j) = sum(filtered_block(:));
        end
    end

    filtered_image = uint8(filtered_image);