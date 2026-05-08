% Generate Patch Antenna Dataset
% Author: Caolán Corrigan
% Student Number: 18478834
%Script to generate dataset for antenna analysis CNN

set(0,'DefaultFigureVisible','off');
clear; clc; close all;

% Global Settings

f0 = 2.4e9;               % 2.4 GHz design frequency
numAnt = 1000;            % Number of antennas
startIdx = 0;             % Begin images and labels at a certain count for adding.

imgFolder = 'images_128x128';
if ~exist(imgFolder, 'dir')
    mkdir(imgFolder);
end

labelFile = 'labels.csv';
fid = fopen(labelFile, 'w');
fprintf(fid, 'filename,L(m),W(m),xFeed(m),yFeed(m),Gmax_dBi,S11_dB,InputResistance_ohm,InputReactance_ohm\n');
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

L_min = 0.86 * L0;
L_max = 1.10 * L0;
W_min = 0.65 * W0;
W_max = 1.35 * W0;

% Main Loop
savedCount = 0;
attempt = 0;

while savedCount < numAnt

    attempt = attempt + 1;

% Random Geometry Sampling
    % A mixture of patch shapes is used to create more gain variation:
    %   - near baseline patches
    %   - narrower patches
    %   - wider patches
    rGeom = rand;

    if rGeom < 0.40
        % Near baseline 
        L = L0 * (0.92 + 0.12*rand);   % 0.92L0 to 1.04L0
        W = W0 * (0.90 + 0.15*rand);   % 0.90W0 to 1.05W0

    elseif rGeom < 0.70
        % Narrower patch 
        L = L0 * (0.86 + 0.14*rand);   % 0.86L0 to 1.00L0
        W = W0 * (0.65 + 0.18*rand);   % 0.65W0 to 0.83W0

    else
        % Wider patch
        L = L0 * (0.95 + 0.15*rand);   % 0.95L0 to 1.10L0
        W = W0 * (1.05 + 0.30*rand);   % 1.05W0 to 1.35W0
    end

    % Safety clamp so the patch remains smaller than the 60 mm ground plane
    L = min(L, 0.90 * 60e-3);
    W = min(W, 0.90 * 60e-3);

    % Feed Offset Sampling
    r = rand;
    if r < 0.35
        % Mostly poor matching: closer to patch centre
        xFrac = 0.04 + (0.24 - 0.04)*rand;
    elseif r < 0.85
        % Useful matching region found experimentally
        xFrac = 0.30 + (0.44 - 0.30)*rand;
    else
        % Near-edge region for extra bad/good variation
        xFrac = 0.44 + (0.48 - 0.44)*rand;
    end

    % Randomly choose left or right side of the patch.
    if rand < 0.5
        xFrac = -xFrac;
    end

    xFeed = xFrac * L;

    % Small y-offset so the feed marker is not always on exactly the same line.
    % Kept small because large y offsets often create failed designs.
    yFeed = (-0.06 + 0.12*rand) * W;

    % Keep feed inside the patch area with a small safety margin.
    xFeed = max(min(xFeed,  0.46*L), -0.46*L);
    yFeed = max(min(yFeed,  0.46*W), -0.46*W);

    % Create Antenna
    ant = patchMicrostrip( ...
        'Length', L, ...
        'Width',  W, ...
        'Substrate', substrate, ...
        'GroundPlaneLength', 60e-3, ...
        'GroundPlaneWidth',  60e-3, ...
        'FeedOffset', [xFeed, yFeed]);

    % S11
    % Also calculate input impedance from S11 using Z0 = 50 ohms.
    try
        s = sparameters(ant, f0);
        s11 = rfparam(s,1,1);
        S11_f0_dB = 20*log10(abs(s11));

        Z0 = 50;
        Zin = Z0 * (1 + s11) / (1 - s11);
        InputResistance_ohm = real(Zin);
        InputReactance_ohm  = imag(Zin);

    catch
        warning('S11 / impedance failed at attempt %d', attempt);
        continue;
    end

    % Radiation Pattern and Gain
    try
        theta_gain = 0:10:180;
        phi_gain   = 0:10:350;

        pat_gain = pattern(ant, f0, theta_gain, phi_gain);
        Gmax_dBi = max(pat_gain(:));

    catch
        warning('Pattern failed at attempt %d', attempt);
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
    savedCount = savedCount + 1;
    imgName = sprintf('ant_%04d.png', savedCount + startIdx);
    imwrite(img, fullfile(imgFolder, imgName));

    % Append Labels
    fidTemp = fopen(labelFile,'a');
    fprintf(fid,'%s,%.6e,%.6e,%.6e,%.6e,%.4f,%.4f,%.4f,%.4f\n', ...
        imgName, L, W, xFeed, yFeed, Gmax_dBi, S11_f0_dB, ...
        InputResistance_ohm, InputReactance_ohm);
    fclose(fid);

    fprintf('Saved %04d/%04d | attempt %04d | S11 %.2f dB | Rin %.2f | Xin %.2f | xFeed %.2f mm | yFeed %.2f mm\n', ...
        savedCount, numAnt, attempt, S11_f0_dB, InputResistance_ohm, InputReactance_ohm, ...
        xFeed*1e3, yFeed*1e3);
end

fprintf('\nDataset generation complete.\n');
fprintf('Saved: %d / %d antennas\n', savedCount, numAnt);
fprintf('Attempts: %d\n', attempt);
fprintf('Images: %s\n', imgFolder);
fprintf('Labels: %s\n', labelFile);