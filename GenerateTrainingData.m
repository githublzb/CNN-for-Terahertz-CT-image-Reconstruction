main(500);

function main(data_num)
h = 512; w =512;
HTheta = linspace(0,180,1000);
downsample_factor1 = 10;
downsample_factor2 = 20;
for id = 1:data_num
    GT = random_ellipse_generator(h, w);
    [HSino, HRecon, LSino1, LTheta1, LRecon1, LSino2, LTheta2, LRecon2] = generate_CT_data(GT, HTheta, downsample_factor1, downsample_factor2);
    file_name = strcat('Data/',num2str(id), '.mat');
    save(file_name, 'GT', 'HSino', 'HTheta', 'HRecon', 'LSino1', 'LTheta1', 'LRecon1', 'LSino2', 'LTheta2', 'LRecon2');
    disp(id);
end
end

function [HSino, HRecon, LSino1, LTheta1, LRecon1, LSino2, LTheta2, LRecon2] = generate_CT_data(img, theta, factor1, factor2)
HSino = radon(img, theta);
HRecon = iradon(HSino, theta, size(img,1));

LSino1 = HSino(:, 1:factor1:end);
LTheta1 = theta(:, 1:factor1:end);
LRecon1 = iradon(LSino1, LTheta1, size(img,1));

LSino2 = HSino(:, 1:factor2:end);
LTheta2 = theta(:, 1:factor2:end);
LRecon2 = iradon(LSino2, LTheta2, size(img,1));
end

function img = random_ellipse_generator(h, w)
[X,Y] = meshgrid(1:w, 1:h);
img = zeros(h,w);
e_num = randi([10,30]);
for i = 1:e_num
    gray_level = randi([-128, 128]);
    alpha = pi.*rand(1);
    center_x = randi([w/4, 3/4*w]);
    center_y = randi([w/4, 3/4*h]);
    e_a = randi([10,40]);
    e_b = randi([20,40]);
    mask = (((X-center_x)*cos(alpha)-(Y-center_y)*sin(alpha))./e_a).^2 + (((X-center_x)*sin(alpha)+(Y-center_y)*cos(alpha))./e_b).^2 < 1;
    %mask_1 = (((X-center_x)*cos(alpha)-(Y-center_y)*sin(alpha))./(e_a+1)).^2 + (((X-center_x)*sin(alpha)+(Y-center_y)*cos(alpha))./(e_b+1)).^2 <= 1;
    img(mask) = img(mask) + gray_level;
    %img(mask_1) = img(mask_1) + 0.3*gray_level;
end
img = conv2(single(img), ones(3)/9, 'same');
end

%sinogram = zeros(729, 1000);
% p = phantom(512);
% sinogram = radon(p, linspace(0,179,1000));
% recon = iradon(sinogram, linspace(0,179,1000), 512);
% theta = linspace(0,180,1000);
% img = random_ellipse_generator(512, 512);
% img_radon = radon(img, theta);
% img_recon = iradon(img_radon, theta, 512);
% downsample_factor = 20;
% img_radon_downsample = img_radon(:, 1:downsample_factor:end);
% downsample_theta = theta(1:downsample_factor:end);
% downsample_recon = iradon(img_radon_downsample, downsample_theta, 512);
% 
% figure
% imagesc(img);
% colormap(gray);
% 
% figure
% subplot(2,2,1);
% imagesc(img_radon);
% colormap(gray);
% 
% subplot(2,2,2);
% imagesc(img_recon);
% colormap(gray);
% 
% subplot(2,2,3);
% imagesc(img_radon_downsample);
% colormap(gray);
% 
% subplot(2,2,4);
% imagesc(downsample_recon);
% colormap(gray);