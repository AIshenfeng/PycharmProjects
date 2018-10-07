import socket

s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
#AF_INTE 表示使用IPV4,SOCK_STREAM 表示是面向流的

#注意connect是一个tuple参数，包括网址和端口
s.connect(('www.sina.com.cn',80))

#发送数据
s.send(b'GET / HTTP/1.1\r\nHost: www.sina.com.cn\r\nConnection: close\r\n\r\n')
#接收数据
buffer=[]
while True:
    d=s.recv(1024) #recv（1024）   每次循环最多接受1024字节
    if d:
        buffer.append(d)
    else:
        break

data=b''.join(buffer)   #把buffer中的字节串拼接起来，并赋值给data

s.close()               #关闭连接

#返回数据中包含HTTP头和网页本身，把他们分开
header,html=data.split(b'\r\n\r\n',1)
print(header)

#HTML写入文件
with open('sina.html','wb') as f:
    f.write(html)
