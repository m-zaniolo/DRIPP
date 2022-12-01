clc
clear

int = load('gibrSRI24_pers44_sev1.66n_2.txt');
pers = load('gibrSRI24_pers174_sev0.415n_2.txt');
hist = load('gibrSRI24_pers87_sev0.83n_2.txt');
freq = load('gibrSRI24_pers87_sev0.83n_4.txt');

t = datetime(1,1,0)+calmonths(0:1200-1);
figure;
subplot(411);
s = 2;
bar(t,hist(s,:)); hold on;
id = hist(s,:)<0;
a = bar(t(id),hist(s,id));
a.FaceColor = 'r';
a.EdgeColor = 'r';
ylabel('SRI')
title('Historical: 1P-1I-1F')
ylim([-2,2])

subplot(412);
s = 4;%2
bar(t,pers(s,:)); hold on;
id = pers(s,:)<0;
a = bar(t(id),pers(s,id));
a.FaceColor = 'r';
a.EdgeColor = 'r';
ylabel('SRI')
title('Long Mild: 2P-0.5I-1F')
ylim([-2,2])

subplot(413);
s = 5;%4
bar(t,int(s,:)); hold on;
id = int(s,:)<0;
a = bar(t(id),int(s,id), 'r');
a.FaceColor = 'r';
a.EdgeColor = 'r';
ylim([-2,2])
ylabel('SRI')
title('Short Intense: 0.5P-2I-1F')

subplot(414);
s = 4;
bar(t,freq(s,:)); hold on;
id = freq(s,:)<0;
a = bar(t(id),freq(s,id));
a.FaceColor = 'r';
a.EdgeColor = 'r';
ylabel('SRI')
title('Frequent: 1P-1I-2F')
ylim([-2,2])
xlabel('time')