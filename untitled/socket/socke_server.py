import socket
import threading

#指定链接的类型 为IPV4,面向流
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
#监听 127.0.0.1:9999
s.bind(('127.0.0.1',9999))

s.listen(5)  #参数是最多的连接数
print('Waitting for connection~~~~~')

def tcplink(sock,addr):
    print("新连接")
    sock.send(b'welcome')

    while True:
        d=sock.recv(1024)
        if not d or d.decode('utf-8')=='exit':
            break

        sock.send(('hello! %s'%d.decode('utf-8')).encode('utf-8'))
    sock.close()
    print('closed!')

while True:
    sock,addr=s.accept()  #accept()会等待并返回一个连接
    #创建一个线程处理这个客户端的连接
    t=threading.Thread(target=tcplink,args=(sock,addr))
    t.start()





