%16级量化编码函数
function [ four ]=e_coding(Is,q,a)
four=zeros(1,4);
for k=1:16
    if Is>a+(k-1)*q && Is<=a+k*q
        four=dec2bin(k-1,4);
        four=str2num(four(:))';
    else
    end
end

end


