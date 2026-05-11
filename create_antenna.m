set(0,'DefaultFigureVisible','off');
clear; clc; close all;

% Settings
f0 = 2.4e9;   % 2.4 GHz
z0 = 50;      % reference impedance

imgFolder = 'C:\Users\keelo\antenna_cnn\single_test_image';
if ~exist(imgFolder, 'dir')
    mkdir(imgFolder);
end

% Substrate
substrate = dielectric('Name','FR4', ...
                       'EpsilonR',4.4, ...
                       'LossTangent',0.02, ...
                       'Thickness',1.6e-3);

% Baseline patch
basePatch = patchMicrostrip('Substrate', substrate, ...
                            'GroundPlaneLength', 60e-3, ...
                            'GroundPlaneWidth',  60e-3);

basePatch = design(basePatch, f0);

L0 = basePatch.Length;
W0 = basePatch.Width;

fprintf('Baseline patch @ %.2f GHz\n', f0/1e9);
fprintf('L0 = %.3f mm | W0 = %.3f mm\n', L0*1e3, W0*1e3);

L = 1.00 * L0;         % patch length
W = 1.00 * W0;         % patch width
xFeed = -0.004;        % metres
yFeed =  0.001;        % metres

% Create antenna
ant = patchMicrostrip( ...
    'Length', L, ...
    'Width',  W, ...
    'Substrate', substrate, ...
    'GroundPlaneLength', 60e-3, ...
    'GroundPlaneWidth',  60e-3, ...
    'FeedOffset', [xFeed, yFeed]);

% Calculate S11 and input impedance
s = sparameters(ant, f0, z0);
s11 = rfparam(s,1,1);
S11_f0_dB = 20*log10(abs(s11));

Zin = impedance(ant, f0);
Rin_ohm = real(Zin);
Xin_ohm = imag(Zin);

% Calculate maximum gain
theta_gain = 0:10:180;
phi_gain   = 0:10:350;
pat_gain = pattern(ant, f0, theta_gain, phi_gain);
Gmax_dBi = max(pat_gain(:));

fprintf('S11 = %.3f dB\n', S11_f0_dB);
fprintf('Gmax = %.3f dBi\n', Gmax_dBi);
fprintf('Rin = %.3f ohm\n', Rin_ohm);
fprintf('Xin = %.3f ohm\n', Xin_ohm);

% Generate 128x128 geometry image
imgSize = 128;
img = zeros(imgSize, imgSize, 'uint8');   % black background

gpSize = 60e-3;
halfGP = gpSize / 2;
scale = (imgSize - 1) / gpSize;

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

% Border
borderVal = 50;
img(1,:)   = borderVal;
img(end,:) = borderVal;
img(:,1)   = borderVal;
img(:,end) = borderVal;

% Patch fill
fillVal = 220;
img(y1:y2, x1:x2) = fillVal;

% Patch outline
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
imgName = 'new_antenna_1.png';
imwrite(img, fullfile(imgFolder, imgName));

fprintf('\nSaved image: %s\n', fullfile(imgFolder, imgName));
fprintf('Use this file with Python prediction script.\n');

% Optional: show antenna and image
figure;
show(ant);
title('Generated Patch Antenna');

figure;
imshow(img);
title('CNN Input Image');
