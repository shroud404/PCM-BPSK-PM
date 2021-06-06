%%%%%%%%%%%%%%%%%%%%%  PCM-BPSK-PM仿真 %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%  PCM_BPSK_PM.m  %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% by shroud404 2021.5%%%%%%%%%%%%% 
%%%%%程序说明
%%完成了PCM-BPSK-PM单副载波调制解调仿真,观测时域和频域波形
%通信体制具体内容如下：
%码速率1MHz，采样率200MHz
%成型滤波器：升余弦滤波器 滚降指数0.5
%内层调制BPSK 10MHz
%外层调制PM   70MHz
%调制指数Kp 0.5
%幅度A=6
%BPSK使用costas提取相干载波
%PM使用锁相环提取相关载波
%调制方式：PCM-BPSK-PM   编码方式：无
%解调方式：相干解调  译码方式：无
%噪声：线性高斯白噪声
%%%%    仿真环境
%软件版本：MATLAB R2019a
clc
clear
close all;
format long;
tic;
%%
%%%参数定义
Rb=1e6;         %基带（码元）速率1Mbit/s
Fc1=10e6;       %1载频10MHz用于BPSK
Fc2=25e6;       %2载频25MHz用于QPSK
Fc=70e6;        %主载波频70MHz用于PM
fs=200e6;       %采样率满足采样定理200MHz
deltaT=fs/Rb;   %一个码元的样点数，码元宽度=deltat*1/fs
Kp=0.5;         %调相指数：mp=Kp*A
A=6;            %调制信号幅度
%%
%%%生成数据源模拟信号ych1
fm1=100;fm2=200;fm3=300;            %三个正弦波的频率，三个正弦波合成模拟基带信号
x=0:(1*10^-6):10*1/fm1-(1*10^-6);   %样点序列
y1=0.5*sin(2*pi*fm1*x);
y2=0.3*sin(2*pi*fm2*x);
y3=0.4*sin(2*pi*fm3*x);
yCh1=y1+y2+y3;                      %合成模拟信号yCh1
%%%绘制输入信号yCh1波形
figure(1)
subplot(211)
plot(x,yCh1);
xlabel('t/s');ylabel('幅度/V')
title('模拟信号时域波形');
figure(1);
subplot(212)
n = length(yCh1);
yCh1_fft = abs(fftshift(fft(yCh1)*2/n));
f = (-n/2:n/2-1)*(10^6/n);
plot(f,yCh1_fft)
xlabel('频率/Hz')
ylabel('幅度/v')
axis([-800,800,0,0.6])
title('模拟信号频谱')
grid on
%%
%%%模数变换——PCM 13折线编码
sampleVal=8000;%8k抽样率
[sampleData,a13_moddata]=PCM_13Encode(yCh1,1/(1*10^-6),sampleVal);
%%%绘制抽样波形
% figure(2)
% subplot(311)
% dt1=1/sampleVal;
% t1=0:dt1:(length(sampleData)-1)*dt1;
% plot(t1,sampleData);
% title('输入信号yCh1经过PCM抽样后的波形');
% figure(2)
% subplot(312)
% stairs(a13_moddata);                    %形象的画出直方图
% axis([500 600 -2 2]); 
% title('经过PCM编码后的bit数据');
% %%%码型变换得到NRZ-L，将PCM单极性变为双极性NRZ
bipolar_msg_source=a13_moddata*2-1;%单极性变为双极型便于2PSK调制
% figure(2)
% subplot(313)
% stairs(bipolar_msg_source);
% axis([500 600 -2 2]); 
% title('双极性数据');
%%
%对量化数据以fs频率上采样即内插0值
up_bipolar_msg_source=upsample(bipolar_msg_source,deltaT);
%%
%%%成型滤波
%设计升余弦滤波器
rcos_fir = rcosdesign(0.5,6,deltaT);
% figure(3);freqz(rcos_fir)
% fvtool(rcos_fir, 'Analysis', 'impulse')
%对采样后的数据进行升余弦滤波;
rcos_msg_source = filter(rcos_fir,1,up_bipolar_msg_source);
n = length(rcos_msg_source);
rcos_msg_source_fft = abs(fftshift(fft(rcos_msg_source)*2/n));
f = (-n/2:n/2-1)*(fs/n)/10^6;
figure(5);
subplot(211)
plot(rcos_msg_source);
title('成型滤波后时域波形');
subplot(212)
plot(f,rcos_msg_source_fft)
xlabel('频率/MHz')
ylabel('幅度/v')
axis([-3,3,0,0.03])
title('成型滤波后频域波形');
%%%变为复信号
I_Data=rcos_msg_source'; 
Q_Data=zeros(length(rcos_msg_source),1,1);
Signal_Source=I_Data + j*Q_Data; 
%%
%%%BPSK调制载波发送
time = [1:length(rcos_msg_source)];
%time =0:1/fs:(length(a13_moddata)*deltaT-1)/fs;
%载波信号 
Delta_Phase=rand(1)*2*pi;               %随机初相，rad 
Carrier=exp(j*(Fc1*time/fs+Delta_Phase));      %构造载波信号
%调制处理 
rcos_msg_source_carrier = rcos_msg_source.*cos(2*pi*Fc1.*time/fs);
Signal_Channel=Signal_Source.*Carrier'; 
%%波形观察
figure(6);
subplot(211)
plot(rcos_msg_source_carrier);
xlabel('t/s');ylabel('幅度/V')
title('BPSK调制信号时域波形');
subplot(212)
n = length(rcos_msg_source_carrier);
rcos_msg_source_carrier_fft = abs(fftshift(fft(rcos_msg_source_carrier)*2/n));
f = (-n/2:n/2-1)*(fs/n)/10^6;
plot(f,rcos_msg_source_carrier_fft)
xlabel('频率/MHz')
ylabel('幅度/v')
axis([-30,30,0,0.015])
title('BPSK调制信号频域波形');

%%
%%%PM调制
Kp=0.5;  %调相指数：mp=Kp*A
A=1;%调制信号幅度
time =0:1/fs:(length(a13_moddata)*deltaT-1)/fs;
bpsk_pm=A*cos(2*pi*Fc*time+Kp*Signal_Channel');%PM已调信号
bpsk_pm1=A*cos(2*pi*Fc*time+Kp*rcos_msg_source_carrier);%PM已调信号
%%波形观察
figure(7);
subplot(211)
plot(bpsk_pm1);
xlabel('t/s');ylabel('幅度/V')
axis([0,inf,-1.5,1.5])
title('PM-BPSK调制信号时域波形');
subplot(212)
n = length(real(bpsk_pm1));
bpsk_pm1_fft = 20*log10(abs(fftshift(fft(bpsk_pm1))));
f = (-n/2:n/2-1)*(fs/n)/10^6;
plot(f,bpsk_pm1_fft)
xlabel('频率/MHz')
ylabel('幅度/dB')
%axis([-30,30,0,0.015])
title('PM-BPSK调制信号频域波形');
% 
%%
%%%%%%%%%%%%%%%%%  信道    %%%%%%%%%%%%%%%%%%%
%设置信噪比，单位dB
ebn0 =10;
snr = ebn0;
% %%%线性高斯白噪声信道
rcos_msg_source_carrier_addnoise = awgn(bpsk_pm1,snr);
%%波形观察
figure(8);

plot(rcos_msg_source_carrier_addnoise);
xlabel('t/s');ylabel('幅度/V')
title('加噪调制信号时域波形');
% subplot(212)
% n = length(rcos_msg_source_carrier_addnoise);
% rcos_msg_source_carrier_addnoise_fft = (abs(fft(rcos_msg_source_carrier_addnoise)));
% 
% f = (-n/2:n/2-1)*(fs/n)/10^6;
% plot(f,abs(fft(rcos_msg_source_carrier_addnoise)))
% xlabel('频率/MHz')
% ylabel('幅度/dB')
% % axis([-30,30,0,0.015])
% title('加噪调制信号频域波形');
%%
ct=cos(2*pi*Fc*time); %产生载频信号  
st=sin(2*pi*Fc*time); %生成和载波信号相差pi/2的信号
for i=1:length(bpsk_pm)
Ih(i)=bpsk_pm(i)*2*(-ct(i));
Qh(i)=bpsk_pm(i)*2*(st(i));
end
Flp=2*Fc1;
b=fir1(100,Flp/fs,'low'); %设计滤波器
Qm=filter(b,1,Qh); 
Im=filter(b,1,Ih);
figure(21)
m_lpf=20*log(abs(fft(b)))/log(10);
x_f=[0:(fs/length(m_lpf)):fs/2];
plot(x_f,m_lpf(1:length(x_f)))
xlabel('频率/Hz');ylabel('幅度/dB')
Sn(1)=0;
for i=1:length(Qm)  
y(i)=Qm(i)/Im(i);
Sn(i)=atan(y(i));
end
figure(22);
% subplot(211)
plot(real(Sn));
xlabel('t/s');ylabel('幅度/V')
title('PM解调信号时域波形');
% subplot(212)
% n = length(Sn);
% Sn_fft = 20*log10(abs(fftshift(fft(Sn))));
% f = (-n/2:n/2-1)*(fs/n)/10^6;
% plot(f,Sn_fft)
% xlabel('频率/MHz')
% ylabel('幅度/dB')
% %axis([-30,30,0,0.015])
% title('PM解调信号频域波形');


% figure(4);  
% stem(b);  %滤波器单位脉冲响应
%Sn=filter(a,1,Sn); 
%%
L=length(rcos_msg_source);
%%%coatas环设计
%参数清零及初始化
Signal_PLL=zeros(L,1);                  %锁相环锁定及稳定后的数据
NCO_Phase = zeros(L,1);                 %锁定的相位
Discriminator_Out=zeros(L,1);           %鉴相器输出
Freq_Control=zeros(L,1);                %频率控制
PLL_Phase_Part=zeros(L,1);              %锁相环相位响应函数
PLL_Freq_Part=zeros(L,1);               %锁相环频率响应函数
I_PLL = zeros(L,1); 
Q_PLL = zeros(L,1); 
%环路处理 
C1=0.022013;                    %环路滤波器系数C1
C2=0.00024722;                  %环路滤波器系数C2  
for i=2:L 
    Signal_PLL(i)=Sn(i)*exp(-j*mod(NCO_Phase(i-1),2*pi));   %得到环路滤波器前的相乘器的输入
    I_PLL(i)=real(Signal_PLL(i));                                       %环路滤波器前的相乘器的I路输入信息数据
    Q_PLL(i)=imag(Signal_PLL(i));                                       %环路滤波器前的相乘器的Q路输入信息数据
    Discriminator_Out(i)=sign(I_PLL(i))*Q_PLL(i)/abs(Signal_PLL(i));    %鉴相器的输出误差电压信号
    PLL_Phase_Part(i)=Discriminator_Out(i)*C1;                          %环路滤波器对鉴相器输出的误差电压信号处理后得到锁相环相位响应函数
    Freq_Control(i)=PLL_Phase_Part(i)+PLL_Freq_Part(i-1);               %控制压控振荡器的输出信号频率
    PLL_Freq_Part(i)=Discriminator_Out(i)*C2+PLL_Freq_Part(i-1);        %环路滤波器对鉴相器输出的误差电压信号处理后得到锁相环频率响应函数
    NCO_Phase(i)=NCO_Phase(i-1)+Freq_Control(i);                        %压控振荡器进行相位调整
end 
figure(11)
plot(cos(NCO_Phase),'r');grid on        %锁相环提取的载波
hold on 
plot(real(Carrier))                     %发射载波
legend('锁相环提取的载波','发射载波')
figure(12)
%画出锁相环频率响应曲线和相位响应曲线
subplot(2,1,1) 
plot(-PLL_Freq_Part(2:L)*fs); grid on; 
title('锁相环频率响应曲线'); 
axis([0 L -inf inf]); 
subplot(2,1,2) 
plot(PLL_Phase_Part(2:L)*180/pi); 
title('锁相环相位响应曲线'); 
axis([0 L -2 2]); grid on; 

figure(13)
Show_D=2000; %起始位置 
Show_U=4000; %终止位置 
Show_Length=Show_U-Show_D; 
subplot(2,2,1) 
plot(I_Data); grid on; 
title('I路信息数据(调制信号)'); 
axis([1e4 6e4 -0.5 0.5]); 
subplot(2,2,2) 
plot(Q_Data); grid on; 
title('Q路信息数据'); 
%%axis([1 Show_Length -2 2]); 
subplot(2,2,3) 
plot(I_PLL); grid on; 
title('锁相环输出I路信息数据(解调信号)'); 
axis([1e4 6e4 -0.5 0.5]); 
subplot(2,2,4) 
plot(Q_PLL); grid on; 
title('锁相环输出Q路信息数据'); 
%%axis([1 Show_Length -2 2]); 
%%
%%%%%%%%%%%%%%%%%  接收机  %%%%%%%%%%%%%%%%%%%
%%%%%%载波恢复
%%%相干解调
% rcos_msg_source_addnoise =rcos_msg_source_carrier_addnoise.*cos(2*pi*Fc1.*time/fs);
% figure(6);
% subplot(211)
% plot(rcos_msg_source_addnoise);
% title('相干载波相乘后时域波形');
% subplot(212)
% plot(abs(fft(rcos_msg_source_addnoise)));
% title('相干载波相乘后频域波形');
%%%%%%%滤波
%%%%%%匹配滤波
%生成匹配滤波器
rollof_factor =0.5;
rcos_fir = rcosdesign(rollof_factor,6,deltaT);
%滤波
rcos_msg_source_MF = filter(rcos_fir,1,I_PLL');
figure(28);
subplot(211)
plot(rcos_msg_source_MF);
title('匹配滤波后时域波形');
subplot(212)
n = length(rcos_msg_source_MF);
rcos_msg_source_MF_fft = abs(fftshift(fft(rcos_msg_source_MF)*2/n));
f = (-n/2:n/2-1)*(fs/n)/10^6;
plot(f,rcos_msg_source_MF_fft);
axis([-3,3,0,0.2])
xlabel('频率/MHz')
ylabel('幅度/v')
title('匹配滤波后频域波形');
%%%%%最佳采样
%%%选取最佳采样点
decision_site = 1250; %(1200+128+1200)/2 =184 三个滤波器的延迟
%每个符号选取一个点作为判决
rcos_msg_source_MF_option = rcos_msg_source_MF(decision_site:deltaT:end);
%涉及到三个滤波器，固含有滤波器延迟累加

%%%判决
msg_source_MF_option_sign= [sign(rcos_msg_source_MF_option)];
figure(9);
stairs(msg_source_MF_option_sign);
axis([500 600 -2 2]); 
title('判决结果');
%%%%%%%%%%%%%%%%%   信宿    %%%%%%%%%%%%%%%%%%%%
%%%误码率性能比对
%[err_number,bit_err_ratio]=biterr(x,y)
[err_number,bit_err_ratio]=biterr(a13_moddata(1:length(msg_source_MF_option_sign)),(msg_source_MF_option_sign+1)/2);
bit_err_ratio
%%删除样点，对齐数据
len1=length(a13_moddata);%%原始数据样点数
len2=length(msg_source_MF_option_sign);%%判决后恢复样点数
len3=8-mod(len1-len2,8);%%计算尾部多余样点
msg_source_MF_option_sign=msg_source_MF_option_sign(1:length(msg_source_MF_option_sign)-len3);%%删除尾部多余样点
[outData] = PCM_13Decode((msg_source_MF_option_sign+1)/2);
figure(10)
subplot(221)
dt1=1/sampleVal;
t1=0:dt1:(length(sampleData)-1)*dt1;
plot(t1,sampleData);
title('输入信号抽样后的波形');
xlabel('时间/s');ylabel('幅值/V');
subplot(222)           
NFFT=length(sampleData);
freq=fft(sampleData,NFFT)*2/NFFT;
freq_d=abs(fftshift(freq));
w=(-NFFT/2:1:NFFT/2-1)*8000/NFFT; %双边
plot(w,freq_d);
title('输入信号频谱');xlabel('频率/Hz');ylabel('幅值/V');

figure(10)
subplot(223)
t2=0:dt1:(length(outData)-1)*dt1;
plot(t2,outData);
title('解码还原后输出的模拟信号波形');
xlabel('时间/s');ylabel('幅值/V');
subplot(224)           
NFFT=length(sampleData);
freq=fft(outData,NFFT)*2/NFFT;
freq_d=abs(fftshift(freq));
w=(-NFFT/2:1:NFFT/2-1)*8000/NFFT; %双边
plot(w,freq_d);
title('解码还原后输出的模拟信号频谱');xlabel('频率/Hz');ylabel('幅值/V');
toc;



