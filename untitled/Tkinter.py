from tkinter import *
import tkinter.messagebox as messagebox

class Application(Frame):
    def __init__(self,master=None):
        Frame.__init__(self,master)
        self.pack()
        self.createWidgets()

    def createWidgets(self):
        self.nameInput=Entry(self)
        self.nameInput.pack()
        self.helloButton=Button(self,text='Show',command=self.hello)
        self.helloButton.pack()
    def hello(self):
        name=self.nameInput.get() or 'world'
        print(type(name))
        messagebox.showinfo('Message','Hello %s'%name)


App=Application()
App.master.title('hello')
App.mainloop()