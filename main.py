class ParentClass1:
    def foo(self):
        print("ParentClass1.foo()")

class ParentClass2:
    def foo(self):
        print("ParentClass2.foo()")
        
class SubClass1(ParentClass1):
    def bar(self):
        self.foo()
        print("SubClass1.bar()")
        
class SubClass2(ParentClass2):
    def bar(self):
        self.foo()
        print("SubClass2.bar()")
        
        
if __name__ == '__main__':
    obj1 = SubClass1()
    obj2 = SubClass2()

    obj1.bar()    
    obj2.bar()
    obj2.__class__ = SubClass1  # type: ignore
    obj2.bar()