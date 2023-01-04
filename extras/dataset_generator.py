# This script creates a dataset containing temperature values both in Celsius and Farenheit.

import random

# Create a dataset with 1000 temperature values
with open("./dataset.csv", "w") as file:
    file.write("Farenheit,Celsius\n")
    for i in range(350):
        c_temperature = random.randint(-100, 100)
        file.write(f'{c_temperature * 9 / 5 + 32},{c_temperature}\n')
