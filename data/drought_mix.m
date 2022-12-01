clc
clear

n = [2,2,2,5];
sev = -[0.83, 0.83/2, 0.83*2, 0.83 ];
pers = [87, 87*2, 88/2, 87 ];
series = {'all_cachuma', 'mission', 'gibr', 'gibrSRI12', 'gibrSRI24', 'gibrSRI36', 'all_swp'};


for s = 1:length(series)
    mix_cali = [];
    mix_vali = [];
    for scen = 1:4
        str = [series{s} ,'_pers', num2str(pers(scen)), '_sev', num2str(abs(sev(scen))),'n_',num2str(n(scen)),'.txt'];
        a = load(str);
        mix_cali = [mix_cali; a(1:20,:)];
        mix_vali = [mix_vali; a(21:50,:)];
    end
string_cali = ['mix_', series{s}, '_cali.txt'];
dlmwrite(string_cali, mix_cali, 'delimiter', '\t');
string_vali = ['mix_', series{s}, '_vali.txt'];
dlmwrite(string_vali, mix_vali, 'delimiter', '\t');
end