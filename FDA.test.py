# test for the FDA.py module
import unittest
from FDA import list_tools

def classify_problem(input_string):
    given = list_tools()[0](input_string)
    print(given)
    return given

test_matrix = [
    "64,39,None,135,128,None,0.05,None,None,None,None,None", "One Sample Mean",

]



if __name__ == '__main__':
    for i in test_matrix:
        result = classify_problem(i[0])
        if i[1] in result:
            print("Test Passed for: " + i[1])
        else:
            print("Test Failed for: " + i[1]")
