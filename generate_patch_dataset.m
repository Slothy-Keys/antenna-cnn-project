% Generate Patch Antenna Dataset
% Author: Caolán Corrigan
% Student Number: 18478834

set(0,'DefaultFigureVisible','off');
clear; clc; close all;

% Global Settings

f0 = 2.4e9;              % 2.4 GHz design frequency
N  = 100;                % Number of antennas
startIdx = 900;          % Begin images and labels at a certain count for adding.

imgFolder = 'images_128x128';
if ~exist(imgFolder, 'dir')
    mkdir(imgFolder);
end

labelFile = 'labels.csv';
fid = fopen(labelFile, 'w');
fprintf(fid, 'filename,L(m),W(m),xFeed(m),yFeed(m),Gmax_dBi,S11_dB,thetaMain_deg\n');
fclose(fid);

% Substrate + Baseline Design

substrate = dielectric('Name','FR4', ...
                       'EpsilonR',4.4, ...
                       'LossTangent',0.02, ...
                       'Thickness',1.6e-3);

basePatch = patchMicrostrip('Substrate',substrate, ...
                            'GroundPlaneLength',60e-3, ...
                            'GroundPlaneWidth',60e-3);

basePatch = design(basePatch, f0);

L0 = basePatch.Length;
W0 = basePatch.Width;

fprintf('Baseline patch @ %.2f GHz\n', f0/1e9);
fprintf('L0 = %.3f mm | W0 = %.3f mm\n', L0*1e3, W0*1e3);

L_min = 0.8 * L0;
L_max = 1.2 * L0;
W_min = 0.8 * W0;
W_max = 1.2 * W0;

% Parallel Pool

if isempty(gcp('nocreate'))
    parpool;
end

% Main Parallel Loop

parfor i = 1:N

    % Random Geometry Sampling
    L = L_min + (L_max - L_min)*rand;
    W = W_min + (W_max - W_min)*rand;

    % Feed offset limits
    maxFeedAllowed  = 0.029;
    maxFeedPhysical = 0.3 * L;
    maxFeed = min(maxFeedAllowed, maxFeedPhysical);

    xFeed = (rand*2 - 1) * maxFeed;

    % Small y-offset for more dataset diversity
    maxYFeedPhysical = 0.15 * W;
    yFeed = (rand*2 - 1) * maxYFeedPhysical;

    % Create Antenna
    ant = patchMicrostrip( ...
        'Length', L, ...
        'Width',  W, ...
        'Substrate', substrate, ...
        'GroundPlaneLength', 60e-3, ...
        'GroundPlaneWidth',  60e-3, ...
        'FeedOffset', [xFeed, yFeed]);

    % S11 at Single Frequency
    try
        s = sparameters(ant, f0);
        s11 = rfparam(s,1,1);
        S11_f0_dB = 20*log10(abs(s11));
    catch
        warning('S11 failed at sample %d', i);
        continue;
    end

  % Radiation Pattern and Gain
    try
        %Gain: coarse 2D global search
        theta_gain = 0:10:180;
        phi_gain   = 0:10:350;

        pat_gain = pattern(ant, f0, theta_gain, phi_gain);
        Gmax_dBi = max(pat_gain(:));

        % Calculate theta
        theta_theta = 0:1:180;
        phi_theta   = 0;

        pat_theta = pattern(ant, f0, theta_theta, phi_theta);
        [~, idxTheta] = max(pat_theta);
        thetaMain = theta_theta(idxTheta);

    catch
        warning('Pattern failed at sample %d', i);
        continue;
    end

    % Generate 128x128 Geometry Image
    imgSize = 128;
    img = zeros(imgSize, imgSize, 'uint8');   % black background

    % Physical ground plane size
    gpSize = 60e-3;
    halfGP = gpSize / 2;

    % Scale metres -> pixels
    scale = (imgSize - 1) / gpSize;

    % Convert physical coordinate (m) to pixel index
    physToPix = @(v) round((v + halfGP) * scale) + 1;

    % Patch corners in metres
    x1m = -L/2;  x2m = L/2;
    y1m = -W/2;  y2m = W/2;

    % Convert to pixel coordinates
    x1 = physToPix(x1m);
    x2 = physToPix(x2m);
    y1 = physToPix(y1m);
    y2 = physToPix(y2m);

    % Clamp indices
    x1 = max(1, min(imgSize, x1));
    x2 = max(1, min(imgSize, x2));
    y1 = max(1, min(imgSize, y1));
    y2 = max(1, min(imgSize, y2));

    % Ensure ordering
    if x1 > x2
        temp = x1; x1 = x2; x2 = temp;
    end
    if y1 > y2
        temp = y1; y1 = y2; y2 = temp;
    end

    % Draw faint substrate
    borderVal = 50;
    img(1,:)   = borderVal;
    img(end,:) = borderVal;
    img(:,1)   = borderVal;
    img(:,end) = borderVal;

    % Fill patch
    fillVal = 220;
    img(y1:y2, x1:x2) = fillVal;

    % Add patch outline for stronger edge visibility
    edgeVal = 255;
    img(y1:y2, x1) = edgeVal;
    img(y1:y2, x2) = edgeVal;
    img(y1, x1:x2) = edgeVal;
    img(y2, x1:x2) = edgeVal;

    % Feed point
    fx = physToPix(xFeed);
    fy = physToPix(yFeed);

    if fx >= 1 && fx <= imgSize && fy >= 1 && fy <= imgSize
        feedRadius = 2;  % 5x5 marker
        fx1 = max(1, fx - feedRadius);
        fx2 = min(imgSize, fx + feedRadius);
        fy1 = max(1, fy - feedRadius);
        fy2 = min(imgSize, fy + feedRadius);

        feedVal = 128;
        img(fy1:fy2, fx1:fx2) = feedVal;
    end

    % Save image
    imgName = sprintf('ant_%04d.png', i + startIdx);
    imwrite(img, fullfile(imgFolder, imgName));

    % Append Labels
    tempFile = sprintf('temp_label_%04d.txt', i + startIdx);
    fidTemp = fopen(tempFile,'w');
    fprintf(fidTemp,'%s,%.6e,%.6e,%.6e,%.6e,%.4f,%.4f,%.2f\n', ...
        imgName, L, W, xFeed, yFeed, Gmax_dBi, S11_f0_dB, thetaMain);
    fclose(fidTemp);

    if mod(i,20)==0
        fprintf('Generated %d / %d\n', i, N);
    end
end

% Merge Temporary Label Files
fid = fopen(labelFile,'a');
for i = 1:N
    tempFile = sprintf('temp_label_%04d.txt', i + startIdx);
    if exist(tempFile,'file')
        txt = fileread(tempFile);
        fprintf(fid,'%s',txt);
        delete(tempFile);
    end
end
fclose(fid);

fprintf('\nDataset generation complete.\n');
fprintf('Images: %s\n', imgFolder);
fprintf('Labels: %s\n', labelFile);