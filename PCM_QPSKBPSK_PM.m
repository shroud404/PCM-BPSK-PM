%%%%%%%%%%%%%%%%%%%%%  PCM-BPSK-PM���� %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%  PCM_BPSK_PM.m  %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% by shroud404 2021.5%%%%%%%%%%%%% 
%%%%%����˵��
%%�����PCM-BPSK-PM�����ز����ƽ������,�۲�ʱ���Ƶ����
%ͨ�����ƾ����������£�
%������1MHz��������200MHz
%�����˲������������˲��� ����ָ��0.5
%�ڲ����BPSK 10MHz
%������PM   70MHz
%����ָ��Kp 0.5
%����A=6
%BPSKʹ��costas��ȡ����ز�
%PMʹ�����໷��ȡ����ز�
%���Ʒ�ʽ��PCM-BPSK-PM   ���뷽ʽ����
%�����ʽ����ɽ��  ���뷽ʽ����
%���������Ը�˹������
%%%%    ���滷��
%����汾��MATLAB R2019a
clc
clear
close all;
format long;
tic;
%%
%%%��������
Rb=1e6;         %��������Ԫ������1Mbit/s
Fc1=10e6;       %1��Ƶ10MHz����BPSK
Fc2=25e6;       %2��Ƶ25MHz����QPSK
Fc=70e6;        %���ز�Ƶ70MHz����PM
fs=200e6;       %�����������������200MHz
deltaT=fs/Rb;   %һ����Ԫ������������Ԫ���=deltat*1/fs
Kp=0.5;         %����ָ����mp=Kp*A
A=6;            %�����źŷ���
%%
%%%��������Դģ���ź�ych1
fm1=100;fm2=200;fm3=300;            %�������Ҳ���Ƶ�ʣ��������Ҳ��ϳ�ģ������ź�
x=0:(1*10^-6):10*1/fm1-(1*10^-6);   %��������
y1=0.5*sin(2*pi*fm1*x);
y2=0.3*sin(2*pi*fm2*x);
y3=0.4*sin(2*pi*fm3*x);
yCh1=y1+y2+y3;                      %�ϳ�ģ���ź�yCh1
%%%���������ź�yCh1����
figure(1)
subplot(211)
plot(x,yCh1);
xlabel('t/s');ylabel('����/V')
title('ģ���ź�ʱ����');
figure(1);
subplot(212)
n = length(yCh1);
yCh1_fft = abs(fftshift(fft(yCh1)*2/n));
f = (-n/2:n/2-1)*(10^6/n);
plot(f,yCh1_fft)
xlabel('Ƶ��/Hz')
ylabel('����/v')
axis([-800,800,0,0.6])
title('ģ���ź�Ƶ��')
grid on
%%
%%%ģ���任����PCM 13���߱���
sampleVal=8000;%8k������
[sampleData,a13_moddata]=PCM_13Encode(yCh1,1/(1*10^-6),sampleVal);
%%%���Ƴ�������
% figure(2)
% subplot(311)
% dt1=1/sampleVal;
% t1=0:dt1:(length(sampleData)-1)*dt1;
% plot(t1,sampleData);
% title('�����ź�yCh1����PCM������Ĳ���');
% figure(2)
% subplot(312)
% stairs(a13_moddata);                    %����Ļ���ֱ��ͼ
% axis([500 600 -2 2]); 
% title('����PCM������bit����');
% %%%���ͱ任�õ�NRZ-L����PCM�����Ա�Ϊ˫����NRZ
bipolar_msg_source=a13_moddata*2-1;%�����Ա�Ϊ˫���ͱ���2PSK����
% figure(2)
% subplot(313)
% stairs(bipolar_msg_source);
% axis([500 600 -2 2]); 
% title('˫��������');
%%
%������������fsƵ���ϲ������ڲ�0ֵ
up_bipolar_msg_source=upsample(bipolar_msg_source,deltaT);
%%
%%%�����˲�
%����������˲���
rcos_fir = rcosdesign(0.5,6,deltaT);
% figure(3);freqz(rcos_fir)
% fvtool(rcos_fir, 'Analysis', 'impulse')
%�Բ���������ݽ����������˲�;
rcos_msg_source = filter(rcos_fir,1,up_bipolar_msg_source);
n = length(rcos_msg_source);
rcos_msg_source_fft = abs(fftshift(fft(rcos_msg_source)*2/n));
f = (-n/2:n/2-1)*(fs/n)/10^6;
figure(5);
subplot(211)
plot(rcos_msg_source);
title('�����˲���ʱ����');
subplot(212)
plot(f,rcos_msg_source_fft)
xlabel('Ƶ��/MHz')
ylabel('����/v')
axis([-3,3,0,0.03])
title('�����˲���Ƶ����');
%%%��Ϊ���ź�
I_Data=rcos_msg_source'; 
Q_Data=zeros(length(rcos_msg_source),1,1);
Signal_Source=I_Data + j*Q_Data; 
%%
%%%BPSK�����ز�����
time = [1:length(rcos_msg_source)];
%time =0:1/fs:(length(a13_moddata)*deltaT-1)/fs;
%�ز��ź� 
Delta_Phase=rand(1)*2*pi;               %������࣬rad 
Carrier=exp(j*(Fc1*time/fs+Delta_Phase));      %�����ز��ź�
%���ƴ��� 
rcos_msg_source_carrier = rcos_msg_source.*cos(2*pi*Fc1.*time/fs);
Signal_Channel=Signal_Source.*Carrier'; 
%%���ι۲�
figure(6);
subplot(211)
plot(rcos_msg_source_carrier);
xlabel('t/s');ylabel('����/V')
title('BPSK�����ź�ʱ����');
subplot(212)
n = length(rcos_msg_source_carrier);
rcos_msg_source_carrier_fft = abs(fftshift(fft(rcos_msg_source_carrier)*2/n));
f = (-n/2:n/2-1)*(fs/n)/10^6;
plot(f,rcos_msg_source_carrier_fft)
xlabel('Ƶ��/MHz')
ylabel('����/v')
axis([-30,30,0,0.015])
title('BPSK�����ź�Ƶ����');

%%
%%%PM����
Kp=0.5;  %����ָ����mp=Kp*A
A=1;%�����źŷ���
time =0:1/fs:(length(a13_moddata)*deltaT-1)/fs;
bpsk_pm=A*cos(2*pi*Fc*time+Kp*Signal_Channel');%PM�ѵ��ź�
bpsk_pm1=A*cos(2*pi*Fc*time+Kp*rcos_msg_source_carrier);%PM�ѵ��ź�
%%���ι۲�
figure(7);
subplot(211)
plot(bpsk_pm1);
xlabel('t/s');ylabel('����/V')
axis([0,inf,-1.5,1.5])
title('PM-BPSK�����ź�ʱ����');
subplot(212)
n = length(real(bpsk_pm1));
bpsk_pm1_fft = 20*log10(abs(fftshift(fft(bpsk_pm1))));
f = (-n/2:n/2-1)*(fs/n)/10^6;
plot(f,bpsk_pm1_fft)
xlabel('Ƶ��/MHz')
ylabel('����/dB')
%axis([-30,30,0,0.015])
title('PM-BPSK�����ź�Ƶ����');
% 
%%
%%%%%%%%%%%%%%%%%  �ŵ�    %%%%%%%%%%%%%%%%%%%
%��������ȣ���λdB
ebn0 =10;
snr = ebn0;
% %%%���Ը�˹�������ŵ�
rcos_msg_source_carrier_addnoise = awgn(bpsk_pm1,snr);
%%���ι۲�
figure(8);

plot(rcos_msg_source_carrier_addnoise);
xlabel('t/s');ylabel('����/V')
title('��������ź�ʱ����');
% subplot(212)
% n = length(rcos_msg_source_carrier_addnoise);
% rcos_msg_source_carrier_addnoise_fft = (abs(fft(rcos_msg_source_carrier_addnoise)));
% 
% f = (-n/2:n/2-1)*(fs/n)/10^6;
% plot(f,abs(fft(rcos_msg_source_carrier_addnoise)))
% xlabel('Ƶ��/MHz')
% ylabel('����/dB')
% % axis([-30,30,0,0.015])
% title('��������ź�Ƶ����');
%%
ct=cos(2*pi*Fc*time); %������Ƶ�ź�  
st=sin(2*pi*Fc*time); %���ɺ��ز��ź����pi/2���ź�
for i=1:length(bpsk_pm)
Ih(i)=bpsk_pm(i)*2*(-ct(i));
Qh(i)=bpsk_pm(i)*2*(st(i));
end
Flp=2*Fc1;
b=fir1(100,Flp/fs,'low'); %����˲���
Qm=filter(b,1,Qh); 
Im=filter(b,1,Ih);
figure(21)
m_lpf=20*log(abs(fft(b)))/log(10);
x_f=[0:(fs/length(m_lpf)):fs/2];
plot(x_f,m_lpf(1:length(x_f)))
xlabel('Ƶ��/Hz');ylabel('����/dB')
Sn(1)=0;
for i=1:length(Qm)  
y(i)=Qm(i)/Im(i);
Sn(i)=atan(y(i));
end
figure(22);
% subplot(211)
plot(real(Sn));
xlabel('t/s');ylabel('����/V')
title('PM����ź�ʱ����');
% subplot(212)
% n = length(Sn);
% Sn_fft = 20*log10(abs(fftshift(fft(Sn))));
% f = (-n/2:n/2-1)*(fs/n)/10^6;
% plot(f,Sn_fft)
% xlabel('Ƶ��/MHz')
% ylabel('����/dB')
% %axis([-30,30,0,0.015])
% title('PM����ź�Ƶ����');


% figure(4);  
% stem(b);  %�˲�����λ������Ӧ
%Sn=filter(a,1,Sn); 
%%
L=length(rcos_msg_source);
%%%coatas�����
%�������㼰��ʼ��
Signal_PLL=zeros(L,1);                  %���໷�������ȶ��������
NCO_Phase = zeros(L,1);                 %��������λ
Discriminator_Out=zeros(L,1);           %���������
Freq_Control=zeros(L,1);                %Ƶ�ʿ���
PLL_Phase_Part=zeros(L,1);              %���໷��λ��Ӧ����
PLL_Freq_Part=zeros(L,1);               %���໷Ƶ����Ӧ����
I_PLL = zeros(L,1); 
Q_PLL = zeros(L,1); 
%��·���� 
C1=0.022013;                    %��·�˲���ϵ��C1
C2=0.00024722;                  %��·�˲���ϵ��C2  
for i=2:L 
    Signal_PLL(i)=Sn(i)*exp(-j*mod(NCO_Phase(i-1),2*pi));   %�õ���·�˲���ǰ�������������
    I_PLL(i)=real(Signal_PLL(i));                                       %��·�˲���ǰ���������I·������Ϣ����
    Q_PLL(i)=imag(Signal_PLL(i));                                       %��·�˲���ǰ���������Q·������Ϣ����
    Discriminator_Out(i)=sign(I_PLL(i))*Q_PLL(i)/abs(Signal_PLL(i));    %���������������ѹ�ź�
    PLL_Phase_Part(i)=Discriminator_Out(i)*C1;                          %��·�˲����Լ��������������ѹ�źŴ����õ����໷��λ��Ӧ����
    Freq_Control(i)=PLL_Phase_Part(i)+PLL_Freq_Part(i-1);               %����ѹ������������ź�Ƶ��
    PLL_Freq_Part(i)=Discriminator_Out(i)*C2+PLL_Freq_Part(i-1);        %��·�˲����Լ��������������ѹ�źŴ����õ����໷Ƶ����Ӧ����
    NCO_Phase(i)=NCO_Phase(i-1)+Freq_Control(i);                        %ѹ������������λ����
end 
figure(11)
plot(cos(NCO_Phase),'r');grid on        %���໷��ȡ���ز�
hold on 
plot(real(Carrier))                     %�����ز�
legend('���໷��ȡ���ز�','�����ز�')
figure(12)
%�������໷Ƶ����Ӧ���ߺ���λ��Ӧ����
subplot(2,1,1) 
plot(-PLL_Freq_Part(2:L)*fs); grid on; 
title('���໷Ƶ����Ӧ����'); 
axis([0 L -inf inf]); 
subplot(2,1,2) 
plot(PLL_Phase_Part(2:L)*180/pi); 
title('���໷��λ��Ӧ����'); 
axis([0 L -2 2]); grid on; 

figure(13)
Show_D=2000; %��ʼλ�� 
Show_U=4000; %��ֹλ�� 
Show_Length=Show_U-Show_D; 
subplot(2,2,1) 
plot(I_Data); grid on; 
title('I·��Ϣ����(�����ź�)'); 
axis([1e4 6e4 -0.5 0.5]); 
subplot(2,2,2) 
plot(Q_Data); grid on; 
title('Q·��Ϣ����'); 
%%axis([1 Show_Length -2 2]); 
subplot(2,2,3) 
plot(I_PLL); grid on; 
title('���໷���I·��Ϣ����(����ź�)'); 
axis([1e4 6e4 -0.5 0.5]); 
subplot(2,2,4) 
plot(Q_PLL); grid on; 
title('���໷���Q·��Ϣ����'); 
%%axis([1 Show_Length -2 2]); 
%%
%%%%%%%%%%%%%%%%%  ���ջ�  %%%%%%%%%%%%%%%%%%%
%%%%%%�ز��ָ�
%%%��ɽ��
% rcos_msg_source_addnoise =rcos_msg_source_carrier_addnoise.*cos(2*pi*Fc1.*time/fs);
% figure(6);
% subplot(211)
% plot(rcos_msg_source_addnoise);
% title('����ز���˺�ʱ����');
% subplot(212)
% plot(abs(fft(rcos_msg_source_addnoise)));
% title('����ز���˺�Ƶ����');
%%%%%%%�˲�
%%%%%%ƥ���˲�
%����ƥ���˲���
rollof_factor =0.5;
rcos_fir = rcosdesign(rollof_factor,6,deltaT);
%�˲�
rcos_msg_source_MF = filter(rcos_fir,1,I_PLL');
figure(28);
subplot(211)
plot(rcos_msg_source_MF);
title('ƥ���˲���ʱ����');
subplot(212)
n = length(rcos_msg_source_MF);
rcos_msg_source_MF_fft = abs(fftshift(fft(rcos_msg_source_MF)*2/n));
f = (-n/2:n/2-1)*(fs/n)/10^6;
plot(f,rcos_msg_source_MF_fft);
axis([-3,3,0,0.2])
xlabel('Ƶ��/MHz')
ylabel('����/v')
title('ƥ���˲���Ƶ����');
%%%%%��Ѳ���
%%%ѡȡ��Ѳ�����
decision_site = 1250; %(1200+128+1200)/2 =184 �����˲������ӳ�
%ÿ������ѡȡһ������Ϊ�о�
rcos_msg_source_MF_option = rcos_msg_source_MF(decision_site:deltaT:end);
%�漰�������˲������̺����˲����ӳ��ۼ�

%%%�о�
msg_source_MF_option_sign= [sign(rcos_msg_source_MF_option)];
figure(9);
stairs(msg_source_MF_option_sign);
axis([500 600 -2 2]); 
title('�о����');
%%%%%%%%%%%%%%%%%   ����    %%%%%%%%%%%%%%%%%%%%
%%%���������ܱȶ�
%[err_number,bit_err_ratio]=biterr(x,y)
[err_number,bit_err_ratio]=biterr(a13_moddata(1:length(msg_source_MF_option_sign)),(msg_source_MF_option_sign+1)/2);
bit_err_ratio
%%ɾ�����㣬��������
len1=length(a13_moddata);%%ԭʼ����������
len2=length(msg_source_MF_option_sign);%%�о���ָ�������
len3=8-mod(len1-len2,8);%%����β����������
msg_source_MF_option_sign=msg_source_MF_option_sign(1:length(msg_source_MF_option_sign)-len3);%%ɾ��β����������
[outData] = PCM_13Decode((msg_source_MF_option_sign+1)/2);
figure(10)
subplot(221)
dt1=1/sampleVal;
t1=0:dt1:(length(sampleData)-1)*dt1;
plot(t1,sampleData);
title('�����źų�����Ĳ���');
xlabel('ʱ��/s');ylabel('��ֵ/V');
subplot(222)           
NFFT=length(sampleData);
freq=fft(sampleData,NFFT)*2/NFFT;
freq_d=abs(fftshift(freq));
w=(-NFFT/2:1:NFFT/2-1)*8000/NFFT; %˫��
plot(w,freq_d);
title('�����ź�Ƶ��');xlabel('Ƶ��/Hz');ylabel('��ֵ/V');

figure(10)
subplot(223)
t2=0:dt1:(length(outData)-1)*dt1;
plot(t2,outData);
title('���뻹ԭ�������ģ���źŲ���');
xlabel('ʱ��/s');ylabel('��ֵ/V');
subplot(224)           
NFFT=length(sampleData);
freq=fft(outData,NFFT)*2/NFFT;
freq_d=abs(fftshift(freq));
w=(-NFFT/2:1:NFFT/2-1)*8000/NFFT; %˫��
plot(w,freq_d);
title('���뻹ԭ�������ģ���ź�Ƶ��');xlabel('Ƶ��/Hz');ylabel('��ֵ/V');
toc;



