%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  FileName:            PCM_13Decode.m
%  Description:         PCM 13���߽���
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Parameter List:       
%       Output Parameter
%           outData     �����ģ������
%       Input Parameter
%           inputData	��������bit������

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function  [outData] = PCM_13Decode( inputData )

n=length(inputData);
outData=zeros(1,n/8);
MM=zeros(1,8);

for kk=1:n/8
    MM(1:8)=inputData(1,(kk-1)*8+1:kk*8); %ȡ��8λPCM��
    temp=MM(2)*2^2+MM(3)*2+MM(4); %��PCM���2~4λת��Ϊ10���ƣ������жϳ���ֵ���ĸ��������� 
    if temp==0
        q=1;    %�������ڷֳ�16�ݣ�ÿһ�ݵĳ���
        a=0;    %�������ڵ���ʼֵ
    end
    if temp==1
        q=1;
        a=16;
    end
    if temp==2
        q=2;
        a=32;
    end
    if temp==3
        q=4;
        a=64;
    end
    if temp==4
        q=8;
        a=128;
    end
    if temp==5
        q=16;
        a=256;
    end
    if temp==6
        q=32;
        a=512;
    end
    if temp==7
        q=64;
        a=1024;
    end
    

    R=( a +( MM(5)*2^3+MM(6)*2^2+MM(7)*2+MM(8)   )*q+q/2) /2048;
    if  MM(1)==0  %�ж�����
        R=-R;
    end
    
    outData(1,kk)=R;
end
end

