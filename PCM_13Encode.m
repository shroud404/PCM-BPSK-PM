%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  FileName:            PCM_13Encode.m
%  Description:         PCM 13折线语音编码
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Parameter List:       
%       Output Parameter
%           sampleData	抽样后数据
%           a13_moddata 编码后的bit流数据
%       Input Parameter
%           inputData	输入模拟数据
%           Fs          模拟信号原始采样率          
%           sampleVal   抽样率
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [sampleData,a13_moddata] = PCM_13Encode(inputData,Fs,sampleVal)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%抽样
sampleData=resample(inputData,sampleVal,Fs);%抽样（降采样），将信号采样率从Fs变为samleVal.
%resample函数是抽取decimate和插值interp的结合.(降采样如果非整数倍，即Fs不能被sampleVal整除，所以需要先插值在抽样)
%先插值sampleVal,信号采样率变为Fs*sampleVal  Hz
%再抽取Fs变成sample Hz


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%13折线编码
[ a13_moddata ] = a_13coding( sampleData );
end





