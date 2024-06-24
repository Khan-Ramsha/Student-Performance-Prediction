from setuptools import find_packages,setup
from typing import List
Hyphen_e="-e ."
def get_requirements(file_path:str)->List[str]:
    # function to get the list of requirements
    requirement=[]
    with open(file_path) as file_obj:
        requirement=file_obj.readlines()
        requirement=[req.replace("\n","") for req in requirement]
        if Hyphen_e in requirement:
            requirement.remove(Hyphen_e)



setup(
name="MLProject",
version="0.0.1",
author="Ramsha",
author_email="khanramsha302020@gmail.com",
packages=find_packages(),
install_requires=get_requirements("requirements.txt")

)