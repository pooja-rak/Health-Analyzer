import pandas as pd
import joblib

model=joblib.load('weight.pkl')
label=joblib.load('label.pkl')

a=int(input("Enter your Age:"))
b=int(input("Enter your Height:"))
c=int(input("Enter your Weight:"))
new = {
    'age': a,
    'height': b,
    'weight': c
}

input_df = pd.DataFrame([new])
prediction = model.predict(input_df)
result = label.inverse_transform(prediction)
print("your results:", result[0])

height_m = b / 100
bmi = c / (height_m ** 2)

min_ideal_weight = 18.5 * (height_m ** 2)
max_ideal_weight = 24.9 * (height_m ** 2)

print(f"\nYou are {a} years old, your height is {b} cm and weight is {c} kg.")
print(f"Your health status: {result}")

if result == 'fit':
    print("You are in a healthy range. Great job!")
else:
    if c < min_ideal_weight:
        print(" You are underweight.")
    elif c > max_ideal_weight:
        print("You are overweight.")
    
    print(f"Ideal weight for your height: {min_ideal_weight:.1f} kg – {max_ideal_weight:.1f} kg")
    
if b > (a * 5 + 80):  # Very rough height-age check
    print(f"\nNote: your height is quite high compared to your age.")
    
