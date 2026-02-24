class Employee:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def __repr__(self):
        return f"Employee(name={self.name}, age={self.age})"

class Department:
    def __init__(self, name: str, employees: list):
        self.name = name
        self.employees = employees

    def show(self):
        print(f"Department: {self.name}")
        for emp in self.employees:
            print(f"  - {emp}")