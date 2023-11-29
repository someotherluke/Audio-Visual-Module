img = imread('cameraman.tif');
zeros_8 = zeros(8, 8);

for i = 1:8;
    for j = 1:8;
        %Determines how big the triangle is
        if j <= 3 - i;
            zeros_8(i, j) = 1;
        end
    end
end

% Define a function handle for the 2D DCT
fun = @(block_struct) zeros_8 .* dct2(block_struct.data);
fun_no_triangle = @(block_struct) dct2(block_struct.data);

inverse_fun = @(block_struct) idct2(block_struct.data);
% Apply block processing to the image using blockproc
B = blockproc(img, [8 8], fun_no_triangle);
C = blockproc(img, [8 8], fun);

B_new = blockproc(B,[8 8], inverse_fun);
C_new = blockproc(C,[8 8], inverse_fun);

figure;
subplot(1,2,1);
imshow(B_new, []);title(['B_new'])
subplot(1,2,2);
imshow(C_new, []);title(['C_new'])
