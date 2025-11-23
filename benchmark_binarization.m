function benchmark_binarization()

    % Carica immagine da txt
    I = load('blobs.txt');
    I = double(I);
    if max(I(:)) <= 1
        I = I * 255;
    end

    window_sizes = [3 5 7 9 11 15 21 31 41];
    num_runs = 10;

    naive_methods = {@sauvola_naive, @niblack_naive, @bernsen_naive, @proposed_naive};
    naive_names   = {'Sauvola Naive', 'Niblack Naive', 'Bernsen Naive', 'Proposed Naive'};
    
    fast_methods  = {@proposed_fast, @otsu_fast};
    fast_names    = {'Proposed Fast', 'Otsu'};

    naive_times = zeros(length(naive_methods), length(window_sizes));
    fast_times  = zeros(length(fast_methods),  length(window_sizes));

    % ===============================
    % BENCHMARK
    % ===============================
    for wi = 1:length(window_sizes)
        w = window_sizes(wi);
        fprintf("\n=== Testing w = %d ===\n", w);

        % ---- NAIVE ----
        for mi = 1:length(naive_methods)
            t = zeros(1, num_runs);
            for r = 1:num_runs
                tic;
                naive_methods{mi}(I, w);
                t(r) = toc;
            end
            naive_times(mi, wi) = mean(t);
            fprintf("%s: %.4f s\n", naive_names{mi}, naive_times(mi,wi));
        end

        % ---- FAST ----
        for mi = 1:length(fast_methods)
            t = zeros(1, num_runs);
            for r = 1:num_runs
                tic;
                fast_methods{mi}(I, w);
                t(r) = toc;
            end
            fast_times(mi, wi) = mean(t);
            fprintf("%s: %.4f s\n", fast_names{mi}, fast_times(mi,wi));
        end
    end

    % ===============================
    % PLOT RISULTATI
    % ===============================
    figure; hold on; grid on;

    % Plot naïve methods (O(n^2 w^2))
    for mi = 1:length(naive_methods)
        scatter(window_sizes, naive_times(mi,:), 60, 'filled', 'DisplayName', naive_names{mi});
    end

    % Plot fast methods (O(n^2))
    for mi = 1:length(fast_methods)
        scatter(window_sizes, fast_times(mi,:), 80, 'filled', 'DisplayName', fast_names{mi});
    end

    set(gca, 'YScale', 'log');
    xlabel("Window size w");
    ylabel("Runtime (seconds, log scale)");
    title("Local vs Global Complexity: O(n^2 w^2) vs O(n^2)");
    legend('Location', 'northwest');
end


% =============================================================
% METODI NAIVE: O(n^2 w^2)
% =============================================================

function B = sauvola_naive(I, w)
    [n,m] = size(I);
    B = zeros(n,m);
    k = 0.34; R = 128;

    d = floor(w/2);
    for x = 1:n
        for y = 1:m
            x1 = max(1, x-d);
            x2 = min(n, x+d);
            y1 = max(1, y-d);
            y2 = min(m, y+1);
            region = I(x1:x2, y1:y2);

            meanR = mean(region(:));
            stdR = std(region(:));

            T = meanR * (1 + k*(stdR/R - 1));
            B(x,y) = I(x,y) > T;
        end
    end
end

function B = niblack_naive(I, w)
    [n,m] = size(I);
    B = zeros(n,m);
    k = -0.2;

    d = floor(w/2);
    for x = 1:n
        for y = 1:m
            x1 = max(1, x-d);
            x2 = min(n, x+d);
            y1 = max(1, y-d);
            y2 = min(m, y+d);
            region = I(x1:x2, y1:y2);

            meanR = mean(region(:));
            stdR = std(region(:));

            T = meanR + k*stdR;
            B(x,y) = I(x,y) > T;
        end
    end
end


function B = bernsen_naive(I, w)
    [n,m] = size(I);
    B = zeros(n,m);
    contrast_threshold = 15;

    d = floor(w/2);
    for x = 1:n
        for y = 1:m
            x1 = max(1, x-d);
            x2 = min(n, x+d);
            y1 = max(1, y-d);
            y2 = min(m, y+d);
            region = I(x1:x2, y1:y2);

            Imax = max(region(:));
            Imin = min(region(:));
            C = Imax - Imin;

            if C < contrast_threshold
                T = mean(I(:));
            else
                T = (double(Imax) + double(Imin)) / 2;
            end

            B(x,y) = I(x,y) > T;
        end
    end
end


function B = proposed_naive(I, w)
    [n,m] = size(I);
    B = zeros(n,m);

    d = floor(w/2);
    k = 0.06;

    for x = 1:n
        for y = 1:m
            x1 = max(1, x-d);
            x2 = min(n, x+d);
            y1 = max(1, y-d);
            y2 = min(m, y+d);
            region = I(x1:x2, y1:y2);

            meanR = mean(region(:));
            dev = I(x,y) - meanR;

            T = meanR * (1 + k*(dev/(1-dev+1e-12)));
            B(x,y) = I(x,y) > T;
        end
    end
end


% =============================================================
% METODI FAST: O(n^2)
% =============================================================

function B = proposed_fast(I, w)
    I = double(I);
    meanI = imboxfilt(I, w);   % local mean
    dev = I - meanI;

    k = 0.06;
    T = meanI .* (1 + k*(dev./(1-dev+1e-12)));
    B = I > T;
end

function B = otsu_fast(I, ~)
    T = graythresh(uint8(I)) * 255;
    B = I > T;
end