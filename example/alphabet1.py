class Car:
    def __init__(self):
        self.hh=True
    def shop(self, name, price):

        if self.hh:
            print(name + "价格为：", price)


if __name__ == '__main__':
    car = Car()

    Car().shop("宝马",1000)