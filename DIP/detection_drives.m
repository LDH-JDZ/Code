function [BW, maskedRGBImage] = detection_drives (Igray)

    % Preprocessing:
    Ismooth = Igray;

    % Edge detection using Canny detector
    BWedges = edge(Ismooth, 'Canny', 0.16);

    % Use bwconncomp to find connected components
    cc = bwconncomp(BWedges); 

    % Keep connected components with a sufficient number of pixels (e.g., greater than 50)
    numPixelsThreshold = 50;
    numPixels = cellfun(@numel, cc.PixelIdxList);
    largeCCIdx = find(numPixels >= numPixelsThreshold);

    % Combine large connected components into a single binary image
    BW = false(size(BWedges));
    for idx = 1:length(largeCCIdx)
        BW(cc.PixelIdxList{largeCCIdx(idx)}) = true;
    end

      % Initialize the figure for visualization
    hFig3D = figure;
    hold on;
    
    [rows, cols] = size(Ismooth);
    [X, Y] = meshgrid(1:cols, 1:rows);

    % Manually perform iterations
    numIterations = 50;
    updateFrequency = 1; % 设置每10次迭代更新一次图形

    for iter = 1:numIterations
        % Perform one iteration of active contour
        BW = activecontour(Ismooth, BW, 5, 'Chan-Vese');

        % Every 10 iterations, capture a snapshot of the segmentation
        if mod(iter, updateFrequency) == 0 || iter == 1
            Z = double(BW)*iter; % Increase Z value to show progression over time
            surf(X, Y, Z, 'EdgeColor', 'none', 'FaceColor', 'interp', 'FaceAlpha', 0.3 + (0.7 * iter/numIterations));
            
            xlabel('X-axis');
            ylabel('Y-axis');
            zlabel('Iteration');
            title('3D Surface Evolution of Contours over Iterations');
            drawnow;
        end
    end
    
    colormap(jet(numIterations)); % Use jet colormap
    view(-45, 30); % Set a good viewing angle
    hold off;

    % ... (rest of the code if needed for further processing or returning results)

end