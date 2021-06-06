%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  FileName:            PCM_13Encode.m
%  Description:         PCM 13������������
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Parameter List:       
%       Output Parameter
%           sampleData	����������
%           a13_moddata ������bit������
%       Input Parameter
%           inputData	����ģ������
%           Fs          ģ���ź�ԭʼ������          
%           sampleVal   ������
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [sampleData,a13_moddata] = PCM_13Encode(inputData,Fs,sampleVal)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%����
sampleData=resample(inputData,sampleVal,Fs);%�������������������źŲ����ʴ�Fs��ΪsamleVal.
%resample�����ǳ�ȡdecimate�Ͳ�ֵinterp�Ľ��.(���������������������Fs���ܱ�sampleVal������������Ҫ�Ȳ�ֵ�ڳ���)
%�Ȳ�ֵsampleVal,�źŲ����ʱ�ΪFs*sampleVal  Hz
%�ٳ�ȡFs���sample Hz


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%13���߱���
[ a13_moddata ] = a_13coding( sampleData );
end





