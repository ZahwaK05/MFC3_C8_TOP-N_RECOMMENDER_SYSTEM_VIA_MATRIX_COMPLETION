function local_patch_image_reconstruction()
    % LOCAL PATCH MATRIX COMPLETION USING LOGDET (better visual quality)

    % Load grayscale tulip image
    img_rgb = imread('flowers.png');
    img = rgb2gray(im2double(img_rgb));
    [m, n] = size(img);

    % Define masked block
    block_size = 80;
    buffer = 20; % surrounding area to aid reconstruction
    row_start = round(m/2 - block_size/2);
    row_end   = round(m/2 + block_size/2);
    col_start = round(n/2 - block_size/2);
    col_end   = round(n/2 + block_size/2);

    % Define the patch boundaries (with buffer)
    r1 = max(1, row_start - buffer);
    r2 = min(m, row_end + buffer);
    c1 = max(1, col_start - buffer);
    c2 = min(n, col_end + buffer);

    patch = img(r1:r2, c1:c2);
    mask_patch = ones(size(patch));
    mask_patch((row_start - r1 + 1):(row_end - r1 + 1), (col_start - c1 + 1):(col_end - c1 + 1)) = 0;

    corrupted_patch = patch .* mask_patch;

    % Run matrix completion on the patch only
    mu = 1e-2;
    gamma = 1.3;
    max_iter = 150;
    tolerance = 1e-6;
    reconstructed_patch = MC_LogDet_v3(corrupted_patch, mask_patch, mu, gamma, tolerance, max_iter);

    % Merge back into original image
    img_filled = img;
    filled_patch = patch;
    filled_patch(mask_patch == 0) = reconstructed_patch(mask_patch == 0);
    img_filled(r1:r2, c1:c2) = filled_patch;

       % Display result
    figure;
    subplot(1, 3, 1); imshow(img); title('Original Image'); axis off;

    % Create blank mask image of same size
    img_masked = img;
    img_masked(r1:r2, c1:c2) = corrupted_patch;
    subplot(1, 3, 2); imshow(img_masked); title('Masked Image'); axis off;

    img_filled = img;
    img_filled(r1:r2, c1:c2) = reconstructed_patch;
    subplot(1, 3, 3); imshow(img_filled); title('Reconstructed Image'); axis off;

end

function [X] = MC_LogDet_v3(X0, M, rho, kappa, toler, maxiter)
    G = X0;
    [m, n] = size(X0);
    Y = X0;
    W = X0;
    Z = zeros(m, n);
    for t = 1:maxiter
        D = W - Z / rho;
        X = X_Solver_first(D, rho / 2);
        X(M == 1) = G(M == 1);
        W = max(X + Z / rho, 0);
        Z = Z + rho * (X - W);
        rho = rho * kappa;
        err = norm(X - X0, 'fro') / norm(X0, 'fro');
        if err <= toler
            break;
        end
        X0 = X;
        fprintf('Iter %d: error = %.6f\n', t, err);
    end
end

function [X] = X_Solver_first(D, rho)
    [U, S, V] = svd(D, 'econ');
    S0 = diag(S);
    r = length(S0);
    rt = zeros(r, 1);
    for t = 1:r
        s = S0(t);
        a = 1;
        b = 1 - s;
        c = 1 / (2 * rho) - s;
        delta = b^2 - 4 * a * c;
        if delta <= 0
            rt(t) = 0;
        else
            rts = sort(roots([a, b, c]));
            if rts(1) * rts(2) <= 0
                rt(t) = rts(2);
            elseif rts(2) < 0
                rt(t) = 0;
            else
                fval = log(1 + rts(2)) + rho * (rts(2) - s)^2;
                if fval > log(1) + rho * (0 - s)^2
                    rt(t) = 0;
                else
                    rt(t) = rts(2);
                end
            end
        end
    end
    S_new = diag(rt);
    X = U * S_new * V';
end