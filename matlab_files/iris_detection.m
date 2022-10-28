%% Clear all/Close all
close all
clear all
%% Iris Detection template subset

% struct to hold grayscale eye images, edge maps, pupil extraction, and
% iris extraction
eyes = struct();
subject = 1; %iris being tested 1-46
eye_pos = 'l'; %eye position (left or right)
switch eye_pos
    case 'l'
        pos = 'left';
    case 'r'
        pos = 'right';
end

% 12 sample images, so loop 12 times
for img = 1:4
    %load in images (all are grayscale bitmaps of size 320x240)
    % from MMU Iris database
    eyes(img).orig = im2gray(imread(strcat('.\training\',string(subject),'\',pos,'\','s',string(subject),eye_pos,'_',string(img),'.bmp')));
    
    %Edge Detection: Canny Edge Detector
    eyes(img).edge = edge(eyes(img).orig,'canny',0.15,3);
    
    %plotting canny edge images
%     figure(2); subplot(3,4,img); imshow(eyes(img).edge);
%     title(strcat('s',string(img),'.bmp'))
    
    %Pupil detection threshold (empirically determined)
    eyes(img).pupil = eyes(img).orig < 50;    
    
    %Circular Hough Transform
    [centers, radii] = imfindcircles(eyes(img).pupil,[6 100]);
    
    % retain the strongest circle (pupil edge)
    coords = centers(1,:);%center = P-by-2 matrix, P = number of circles, x in 1st column, y in 2nd column
    pupil_radius = radii(1);
    
    % start from lower right of pupil edge
    x = round(coords(1))+round(pupil_radius);
    y = round(coords(2))+round(pupil_radius);
    while 1 % traverse radially outward (rightward/downward)
        if eyes(img).edge(y,x) == 1 % iris edge detected
            % calculate radial distance from pupil to iris edge
            % calculate radial distance from center of pupil iris edge
            iris_radius = ((x-coords(1))^2 + (y-coords(2))^2)^0.5;
            break
        end
        x = x+1; y = y+1;
        if x == 240 || y == 320 % if no iris edge was detected
            iris_radius = pupil_radius*2; % set iris radius to 2x pupil radius
            break
        end
    end
    
    %plotting orignal picture with iris and pupil identified
    figure(3); subplot(2,2,img); imshow(eyes(img).orig);
    title(strcat('s',string(subject),eye_pos,'_',string(img),'.bmp'))
    % draw pupil circle
    viscircles(centers(1,:), pupil_radius,'Color','b','LineWidth',0.5);
    % draw iris circle
    viscircles(centers(1,:), iris_radius,'Color','b','LineWidth',0.5);
    
    % extract iris
    % return 2-D grid of coordinates for grayscale eye image
    [xgrid, ygrid] = meshgrid(1:size(eyes(img).orig,2), 1:size(eyes(img).orig,1));
    % filter out anything outside iris and anything inside pupil
    % pixel value = 1 for region of interest, 0 for non-ROI
    mask = ((xgrid-coords(1)).^2 + (ygrid-coords(2)).^2) <= iris_radius.^2 &...
        ((xgrid-coords(1)).^2 + (ygrid-coords(2)).^2) >= pupil_radius.^2;
    % apply mask to original grayscale image to get extracted iris
    eyes(img).iris = eyes(img).orig .* uint8(mask);
    
    % plotting iris extracted
%     figure(4); subplot(3,4,img); imshow(eyes(img).iris);
%     title(strcat('s',string(img),'.bmp'))
    
    %Image enhancement 
    %round estimate values for radiis and center point
    rounded_coords = round(coords);
    round_iris_r = round(iris_radius);
    round_pupil_r = round(pupil_radius);
    
    %Resize image to only include iris
    eyes(img).iris_only = eyes(img).iris(rounded_coords(2)- ...
        round_iris_r:rounded_coords(2)+ round_iris_r, rounded_coords(1)- ...
            round_iris_r:rounded_coords(1)+round_iris_r);
    figure(5); subplot(2,2,img); imshow(eyes(img).iris_only);
    title(strcat('s',string(subject),eye_pos,'_',string(img),'.bmp'))
    
    %Remove extreme pixels(eyelid, eyelash, reflection on eye)
    eyes(img).iris_only = uint8(eyes(img).iris_only > 40) .* ...
        uint8(eyes(img).iris_only < 140) .* eyes(img).iris_only;
    
    figure(6); subplot(2,2,img); imshow(eyes(img).iris_only);
    title(strcat('s',string(subject),eye_pos,'_',string(img),'.bmp'))

    %Increase contrast
    [M,N] = size(eyes(img).iris_only);

    for x=1:M
        for y=1:N
            %ignore thresholded pixels
            if eyes(img).iris_only(y,x) == 0 
                continue
            end
            %remap [51,149] intensities to [0,255]
            eyes(img).iris_only(y,x) = ((double(eyes(img).iris_only(y,x))-50)/(130-50))*255;
        end
    end
    
    figure(7); subplot(2,2,img); imshow(eyes(img).iris_only, []);
    title(strcat('s',string(subject),eye_pos,'_',string(img),'.bmp'))
    
    eyes(img).iris_only = uint8(eyes(img).iris_only);

    %Iris normalization 
    eyes(img).polar = ImToPolar(eyes(img).iris_only, ...
        round_pupil_r/round_iris_r, round_iris_r/round_iris_r, 40, 320);

    figure(8); subplot(2,2,img); imshow(eyes(img).polar, []);
    title(strcat('s',string(subject),eye_pos,'_',string(img),'.bmp'))

    %HOG feature extraction
    [eyes(img).featureVector,hogVisualization] = extractHOGFeatures(uint8(eyes(img).polar));
%     eyes(img).featureVector
    
    figure(9); subplot(2,2,img); imshow(uint8(eyes(img).polar));
    title(strcat('s',string(subject),eye_pos,'_',string(img),'.bmp'))
    hold on; 
    plot(hogVisualization);
    hold off;


    
%     eyes(img).features = detectHarrisFeatures(uint8(eyes(img).polar));
%     im_w_detected = insertMarker(uint8(eyes(img).polar), eyes(img).features, 'circle');
%     figure(9); subplot(3,4,img); imshow(im_w_detected, []);
%     title(strcat('s',string(img),'.bmp'))

end

%% Test images subset

% struct to hold grayscale eye images, edge maps, pupil extraction, and
% iris extraction
eyes_test = struct();
% subject = 1+1;
% eye_pos = 'l';
% switch eye_pos
%     case 'l'
%         pos = 'left';
%     case 'r'
%         pos = 'right';
% end

% 12 sample images, so loop 12 times
for img = 1:1
    %load in images (all are grayscale bitmaps of size 320x240)
    % from MMU Iris database
    eyes_test(img).orig = im2gray(imread(strcat('.\training\',string(subject),'\',pos,'\','t',string(subject),eye_pos,'_',string(img),'.bmp')));
    
    %Edge Detection: Canny Edge Detector
    eyes_test(img).edge = edge(eyes_test(img).orig,'canny',0.15,3);
    
    %plotting canny edge images
%     figure(2); subplot(3,4,img); imshow(eyes(img).edge);
%     title(strcat('s',string(img),'.bmp'))
    
    %Pupil detection threshold (empirically determined)
    eyes_test(img).pupil = eyes_test(img).orig < 50;    
    
    %Circular Hough Transform
    [centers, radii] = imfindcircles(eyes_test(img).pupil,[6 100]);
    
    % retain the strongest circle (pupil edge)
    coords = centers(1,:);%center = P-by-2 matrix, P = number of circles, x in 1st column, y in 2nd column
    pupil_radius = radii(1);
    
    % start from lower right of pupil edge
    x = round(coords(1))+round(pupil_radius);
    y = round(coords(2))+round(pupil_radius);
    while 1 % traverse radially outward (rightward/downward)
        if eyes_test(img).edge(y,x) == 1 % iris edge detected
            % calculate radial distance from pupil to iris edge
            % calculate radial distance from center of pupil iris edge
            iris_radius = ((x-coords(1))^2 + (y-coords(2))^2)^0.5;
            break
        end
        x = x+1; y = y+1;
        if x == 240 || y == 320 % if no iris edge was detected
            iris_radius = pupil_radius*2; % set iris radius to 2x pupil radius
            break
        end
    end
    
    %plotting orignal picture with iris and pupil identified
    figure(10); subplot(1,1,img); imshow(eyes_test(img).orig);
    title(strcat('t',string(subject),eye_pos,'_',string(img),'.bmp'))
    % draw pupil circle
    viscircles(centers(1,:), pupil_radius,'Color','b','LineWidth',0.5);
    % draw iris circle
    viscircles(centers(1,:), iris_radius,'Color','b','LineWidth',0.5);
    
    % extract iris
    % return 2-D grid of coordinates for grayscale eye image
    [xgrid, ygrid] = meshgrid(1:size(eyes_test(img).orig,2), 1:size(eyes_test(img).orig,1));
    % filter out anything outside iris and anything inside pupil
    % pixel value = 1 for region of interest, 0 for non-ROI
    mask = ((xgrid-coords(1)).^2 + (ygrid-coords(2)).^2) <= iris_radius.^2 &...
        ((xgrid-coords(1)).^2 + (ygrid-coords(2)).^2) >= pupil_radius.^2;
    % apply mask to original grayscale image to get extracted iris
    eyes_test(img).iris = eyes_test(img).orig .* uint8(mask);
    
    % plotting iris extracted
%     figure(4); subplot(3,4,img); imshow(eyes(img).iris);
%     title(strcat('s',string(img),'.bmp'))
    
    %Image enhancement 
    %round estimate values for radiis and center point
    rounded_coords = round(coords);
    round_iris_r = round(iris_radius);
    round_pupil_r = round(pupil_radius);
    
    %Resize image to only include iris
    eyes_test(img).iris_only = eyes_test(img).iris(rounded_coords(2)- ...
        round_iris_r:rounded_coords(2)+ round_iris_r, rounded_coords(1)- ...
        round_iris_r:rounded_coords(1)+round_iris_r);
    figure(11); subplot(1,1,img); imshow(eyes_test(img).iris_only);
    title(strcat('t',string(subject),eye_pos,'_',string(img),'.bmp'))
    
    %Remove extreme pixels(eyelid, eyelash, reflection on eye)
    eyes_test(img).iris_only = uint8(eyes_test(img).iris_only > 40) .* ...
    uint8(eyes_test(img).iris_only < 140) .* eyes_test(img).iris_only;
    
    figure(12); subplot(1,1,img); imshow(eyes_test(img).iris_only);
    title(strcat('t',string(subject),eye_pos,'_',string(img),'.bmp'))

    %Increase contrast
    [M,N] = size(eyes_test(img).iris_only);

    for x=1:M
        for y=1:N
            if eyes_test(img).iris_only(y,x) == 0
                continue
            end
            eyes_test(img).iris_only(y,x) = ((double(eyes_test(img).iris_only(y,x))-50)/(130-50))*255;
        end
    end
    
    figure(13); subplot(1,1,img); imshow(eyes_test(img).iris_only, []);
    title(strcat('t',string(subject),eye_pos,'_',string(img),'.bmp'))
    
    eyes_test(img).iris_only = uint8(eyes_test(img).iris_only);

    %Iris normalization 
    eyes_test(img).polar = ImToPolar(eyes_test(img).iris_only, ...
        round_pupil_r/round_iris_r, round_iris_r/round_iris_r, 40, 320);

    figure(14); subplot(1,1,img); imshow(eyes_test(img).polar, []);
    title(strcat('t',string(subject),eye_pos,'_',string(img),'.bmp'))

    [eyes_test(img).featureVector,hogVisualization] = extractHOGFeatures(uint8(eyes_test(img).polar));
%     eyes_test(img).featureVector

    figure(15); subplot(1,1,img); imshow(uint8(eyes_test(img).polar)); 
    title(strcat('t',string(subject),eye_pos,'_',string(img),'.bmp'))
    hold on; 
    plot(hogVisualization);
    hold off;
end

%% Feature matching...

% Eucl_distance = norm(eyes(1).featureVector - eyes_test(1).featureVector);
% pause
%Euclidean distance feature matching
%12 sample images, so loop 12 times
fprintf("Testing (%s) eye for subject (%d)\n", pos, subject);
for img = 1:4
    for img_test = 1:1
        Eucl_distance = norm(eyes(img).featureVector - eyes_test(img_test).featureVector);
        if Eucl_distance < 5.5
            Authenticated(img, img_test) = 1;
            fprintf("Match found for (%s) eye(%d) with (%s) eye_test(%d)\n", pos, img, pos, img_test)
        else
            Authenticated(img, img_test) = 0;
        end
    end
end

for img = 1:4
    if sum(Authenticated(img, :))
    else
        fprintf("Match NOT found for (%s) eye(%d) \n", pos, img)
    end
end
