function analysis_pitchdiscreteness
    %%
    outputdir = './output/';

    dataname = {...
        'Ireland old style', 'Yangguan Sandie', 'Happy Birthday',...
        'English_short', 'Sometimes behave so strangely', 'Vietnamese',...
        'CANYO', 'FIREB', 'KAUAI'...
        };

    datatype = {...
        'Music', 'Music', 'Music',...
        'Speech', 'Speech', 'Speech',...
        'Birdsong', 'Birdsong', 'Birdsong'...
        };

    reffreq = 440;

    %%
    addpath('./lib/');
    K = 8;

    if ~exist(outputdir, 'dir')
        mkdir(outputdir);
    end

    f = figure(1);
    f.Position = [10, 300, 1000, 700];
    clf; cla;
    figure(2);
    clf; cla;
    for i=1:numel(dataname)
        %%
        f0info = readtable(strcat('./data/', dataname{i}, '_f0.csv'));
        f0_cent = 1200.*log2(f0info.Var2./reffreq);
        t_f0 = f0info.Var1;
        
        %%
        onsetinfo = readtable(strcat('./data/seg_', dataname{i}, '.csv'));
        breakinfo = readtable(strcat('./data/break_', dataname{i}, '.csv'));
        [~, t_st, t_ed] = h_ioi(onsetinfo.onset_t, breakinfo.break_t);

        %%
        H = [];
        L = [];
        f0_cent = f0_cent(:)';

        for j=1:numel(t_st)
            [~, idx_st] = min(abs(t_f0 - t_st(j)));
            [~, idx_ed] = min(abs(t_f0 - t_ed(j)));

            f0_cent_j = f0_cent(idx_st:idx_ed);

            idx_st_j = find(~isinf(f0_cent_j), 1, 'first');
            while ~isempty(idx_st_j)
                idx_ed_j = find(isinf(f0_cent_j(idx_st_j:end)), 1, 'first') - 1 + idx_st_j - 1;

                if isempty(idx_ed_j)
                    idx_ed_j = numel(f0_cent_j);
                end
                
                if (idx_ed_j - idx_st_j + 1) > K
                    H(end + 1) = klentropy(f0_cent_j(idx_st_j:idx_ed_j), K);
                    L(end + 1) = t_f0(idx_ed_j) - t_f0(idx_st_j);
                end

                idx_st_j = find(~isinf(f0_cent_j(idx_ed_j + 1:end)), 1, 'first') + idx_ed_j;
            end
        end
        
        if strcmp(datatype{i}, 'Music')
            colorcode = '#0072BD';
        elseif strcmp(datatype{i}, 'Speech')
            colorcode = '#D95319';
        elseif strcmp(datatype{i}, 'Birdsong')
            colorcode = '#EDB120';
        end

        figure(2);
        scatter(normrnd(i, 0.1, [numel(H), 1]), H, 'MarkerEdgeColor', colorcode);
        hold on;
        xlim([-1, 10]);

        writetable(array2table([H(:), L(:)], 'VariableNames', {'Entropy', 'Duration'}), strcat(outputdir, dataname{i}, '_H.csv'),...
            'WriteRowNames', false, 'WriteVariableNames', true);

        %%
        figure(1);
        subplot(3, 3, i);
        scatter(t_f0, f0_cent, 3);

        xlim([0, f0info.Var1(end)]);
        if strcmp(datatype{i}, 'Music') || strcmp(datatype{i}, 'Speech')
            ylim([-4000, 0]);
        elseif strcmp(datatype{i}, 'Birdsong')
            ylim([0, 3600]);
        end
        
        title(dataname{i});
    end
end