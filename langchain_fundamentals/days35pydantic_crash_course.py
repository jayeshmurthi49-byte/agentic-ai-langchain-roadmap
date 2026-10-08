from pydantic import (
    BaseModel,Field,EmailStr,ValidationError,
    field_validator,model_validator,computed_field
)

from typing import Optional,List 

# --- 1. Type hints alone do NOT validate ---
def plain_text(name:str,age:int):
    return f"{name} and {age}"

print("1. plain function:",plain_text("jayesh","Tweenty_two"))

# --- 2. BaseModel: required, optional, defaults, coercion ---
class Student(BaseModel):
    name:str
    age:int
    email:EmailStr
    cgp:float = Field(ge=1,le=10,description="CGPA between 0 and 10")
    city: Optional[str] = None
    skills:List[str] = [] 

s = Student(name="Jayesh",age="32",email="jayesh@example.com",cgp=8.5)
print("\n2. Valid student:",s)
print("type of age",type(s.age)) 

# --- 3. Invalid data -> ValidationError ---
try:
    Student(name="X",age=22,email="not-an-email",cgp=15)
except ValidationError as e:
    print("\n3. Validation errors:")
    print(e) 


# --- 4. field_validator: clean or check ONE field ---
class Employee(BaseModel):
    name:str
    email:EmailStr
    role:str
    salary:int


    @field_validator("name")
    @classmethod
    def title_case_name(cls,v):
        return v.title() 

# --- 5. model_validator: check MULTIPLE fields together ---

    @model_validator(mode="after")
    def inter_salary_limit(self):
        if self.role == "intern" and self.salary > 30000:
            raise ValueError("Intern salary cannot exceed 30000")
        return self 

e = Employee(name="jayesh murthi", email="j@example.com", role="engineer", salary=60000)
print("\n4. field_validator cleaned the name:", e.name) 


try:
    Employee(name="amit",email="a@example.com",role="intern",salary=90000)
except ValidationError as err:
    print("\n5.mode_validation caught",err.errors()[0]["msg"])  



# --- 6. computed_field --- 
class Rectangle(BaseModel):
    width:float
    height:float

    @computed_field
    @property
    def are(self)->float:
        return self.width * self.height

print("\n6. computed_field:",Rectangle(width=4,height=5,).model_dump())  


# --- 7. Nested models ---
class Address(BaseModel):
    city:str
    pincode:str

class person(BaseModel):
    name:str
    address:Address

p = person(name="Jayesh", address={"city": "Mumbai", "pincode": "400001"})
print("\n7. Nested:", p.address.pincode)


# --- 8. Serialization ---
print("\n model_dump():      ",p.model_dump())
print("   model_dump_json",p.model_dump_json()) 