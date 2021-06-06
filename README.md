# PCM-BPSK-PM
The modeming simulation of single carrier composite modulation signal is completed.The modulation method is PCM-BPSK-PM.
This course is designed to simulate the completion of a single carrier composite modulation signal and many in MATLAB
The modulation and deconstuning process of the subcarrier composite modulation signa.

PCM-BPSK-PM调制解调仿真
输入模拟信号；
经过PCM编码，抽样8K，编码方式为A律13折
极性变换，变为NRZ-L基带码流；
采样率200MHz；
通过脉冲成型滤波器进行脉冲成型；
内层调制为BPSK载波10MHZ；
外层调制PM载波70MHz；
可选择AWGN信道加噪处理；
相干解调PM——LPF滤波；
使用costas环解调BPSK，得到恢复载波和解调信号；
通过匹配滤波器同样为升余弦滤波器；
补偿滤波器时延，选择最佳采样点；
抽样判决恢复NRZ-L码流；
DPCM解码恢复模拟信号。


