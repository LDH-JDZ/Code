function [filtered_img] = truncated_filter(input_img, min_radius, max_radius, threshold)
[rows, cols] = size(input_img);
filtered_img = double(input_img);

% 边界扩展处理
img_padded = padarray(input_img, [max_radius, max_radius], 'symmetric', 'both');
img_padded=double(img_padded);
% 确定中心点
center = floor((min_radius+max_radius)/2);
% 用于曲率估计的二阶导数滤波器
dxx = [1 -2 1];
dyy = dxx';
result=[];
result2=[];
for i = 1:rows
    for j = 1:cols
        % 计算二阶导数
        img_window = double(img_padded(i:i+2*max_radius, j:j+2*max_radius));
        dxx_img_window = conv2(img_window, dxx, 'same');
        dyy_img_window = conv2(img_window, dyy, 'same');
        
        % 计算局部曲率
        curvature = sqrt(dxx_img_window(center, center)^2 + dyy_img_window(center, center)^2);
        
        % 确定窗口半径
%         window_radius =min_radius + (max_radius - min_radius) * (1 - curvature/max(max(curvature(:)), epsilon));

  curvature_normalized = 1 / (1 + exp(-curvature));
%  curvature_normalized = tanh(1 * curvature);
 window_radius = round(min_radius + (max_radius - min_radius) * curvature_normalized);

%        window_radius = round(min_radius + (max_radius - min_radius) * (1 - (curvature/max(max(curvature(:)), epsilon))^beta));
        % 提取圆形窗口
        [CC, RR] = meshgrid(1:2*window_radius+1, 1:2*window_radius+1);
        circle_window = sqrt((CC - window_radius - 1).^2 + (RR - window_radius - 1).^2) <= window_radius;
        patch = img_padded(i:i+(2*window_radius), j:j+(2*window_radius)) .* double(circle_window);
         result = [ result,window_radius];
      
        % result2=[ result2,a];

%         %% 进行截断中值滤波
      patch = nonzeros(patch);  % 提取非零元素
        mu = median(patch);
        sigma = std(patch);
        
        if abs(filtered_img(i, j) - mu) > threshold * sigma
            filtered_img(i, j) = mu;
        end
%               filtered_img(i, j) =median(patch);
    end
end
filtered_img = uint8(filtered_img);
end