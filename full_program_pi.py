#This version of the program runs on the Raspberry Pi
#uses tflite to run the model instead of the full version of Tensorflow
#from tensorflow import keras
#import tensorflow as tf
import tflite_runtime.interpreter as tflite
import os
import numpy as np
import cv2
from PIL import Image
import os
from picamera import PiCamera
import time
def initialize_recipes():
    recipies = []

    recipies.append({"name": "Fruit salad", "ingredients": ["Apple", "Kiwi", "Grape"], "URL": "https://foodhero.org/recipes/fruit-salad"})
    recipies.append({"name": "Apple Smoothie", "ingredients": ["Apple"], "URL": "https://foodhero.org/recipes/apple-smoothie"})
    recipies.append({"name": "Cucumber Flavored Water", "ingredients": ["Cucumber"], "URL": "https://foodhero.org/recipes/cucumber-flavored-water"})
    recipies.append({"name": "Orange Juice", "ingredients": ["Orange"], "URL": "https://foodhero.org/recipes/orange-juice"})
    recipies.append({"name": "KIwi and mango Smoothie", "ingredients": ["Kiwi", "Mango"]})
    recipies.append({"name": "Strawberry, Apple, & Banana Smoothie", "ingredients": ["Strawberry", "Apple", "Banana"], "URL": "https://www.bigoven.com/recipe/strawberry-apple-banana-smoothie/1464590"})
    recipies.append({"name": "Pomegranate Pizazz", "ingredients": ["Cucumber", "Orange", "Apple", "Pomegranate", "Lemon"], "URL": "https://juicerecipes.com/recipes/pomegranate-pizazz-94"})
    recipies.append({"name": "Mango-Pineapple Sorbet", "ingredients": ["Banana", "Mango", "Pineapple", "Lemon"], "URL": "https://www.womansday.com/food-recipes/food-drinks/recipes/a10187/mango-pineapple-sorbet-121787/"})
    recipies.append({"name": "Fruit Kebabs", "ingredients": ["Grape", "Strawberry", "Pineapple", "Kiwi"], "URL": "https://www.chabad.org/recipes/recipe_cdo/aid/2832751/jewish/Fruit-Kebabs.htm"})
    recipies.append({"name": "Tomato Cucumber Salad", "ingredients": ["Tomato", "Cucumber", "Onion", "Lemon"], "URL": "https://www.allrecipes.com/recipe/14157/tomato-cucumber-salad/"})
    recipies.append({"name": "Baked Potato 'n Turnip Fries", "ingredients": ["Turnip", "Potato"], "URL": "https://www.hungry-girl.com/recipes/baked-potato-n-turnip-fries"})
    recipies.append({"name": "Oven-roast potatoes", "ingredients": ["Garlic", "Potato"], "URL": "https://www.goodhousekeeping.com/uk/food/recipes/a538065/oven-roast-potatoes/"})
    recipies.append({"name": "Apple & Ginger smoothie", "ingredients": ["Apple", "Ginger"], "URL": "https://www.goodhousekeeping.com/uk/food/recipes/a538065/oven-roast-potatoes/"})
    recipies.append({"name": "Banana Apple Sauce Baby Food", "ingredients": ["Apple", "Banana"], "URL": "https://www.bigoven.com/recipe/strawberry-apple-banana-smoothie/1464590"})
    recipies.append({"name": "Japanese Smashed Cucumbers", "ingredients": ["Cucumber", "Garlic"], "URL": "https://foodchannel.com/recipes/japanese-smashed-cucumbers"})

    return recipies



def take_picture():


    #Camera
    camera=PiCamera()
    camera.resolution=(1080,720)
    camera.rotation=180

    #Folder
    path = "/home/pi/Project/kamerabilder"
    isExist = os.path.exists(path)
    if not isExist:
        os.mkdir(path)
    else:
        print("Exist folder")


    #Function
    fortsett = True

    counter = 0

    while fortsett:

        #time.sleep(5)
        userinput = input("Press 1 to take picture, 2 to close")

        if userinput == "1":

            file_name="/home/pi/Project/kamerabilder/image" + str(counter) +".jpg"
            camera.capture(file_name)

            print("Picture taken")

            counter += 1

        else:
            print("closing program")
            break;

def detect_ingredients():

    def preprocess(folder_path, filename):
        img = np.array(Image.open(folder_path + '/' + filename))
        
        img = img.astype(np.float32) - np.mean(img)
        # Read the image
        #img = cv2.imread(filename)
        
        # Resize the image
        img = cv2.resize(img, (64, 64))
        # Convert the image to float32
        #img = image.astype(np.float32)
        # Normalize the pixel values
        img = img / 255.0
        # Expand dimensions to create a batch of size 1
        #image = np.expand_dims(image, axis=0)
        #img = preprocessing.image.load_img(folder_path, target_size=(64, 64))
        img /= np.std(img)
        img = np.expand_dims(img, axis=0)
        return img

    #model = keras.models.load_model('object_recognition_fruit.tflite')
    interpreter = tflite.Interpreter(model_path='object_recognition_fruit_v7.tflite')
    interpreter.allocate_tensors()


    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Folder path containing the images
    folder_path = 'kamerabilder'


    detected_ingredients = []

    # Iterate through the images in the folder
    print("Starting iteration")
    for filename in os.listdir(folder_path):
        print("File name: ", filename)
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust file extensions as needed

            
            image = preprocess(folder_path, filename)
            #interpreter.set_tensor(input_details, image)
            
            # Set the input tensor
            interpreter.set_tensor(input_details[0]['index'], image)
            #interpreter.set_tensor(input_details, image)

            # Run the inference
            interpreter.invoke()

            # Get the output tensor
            output_tensor = interpreter.get_tensor(output_details[0]['index'])
            #output_tensor = interpreter.get_tensor(output_details)

            # Process the output
            predicted_class = np.argmax(output_tensor)

            print(f"Image: {filename}, Predicted class: {predicted_class}")
            detected_ingredients.append(predicted_class)

    ingredients_codes = {0: 'Apple', 1: 'Banana', 2: 'Cabbage', 3: 'Cucumber', 4: 'Garlic', 5: 'Ginger', 6: 'Grape', 7: 'Kiwi', 8: "Orange",
                          9: "Bell Pepper"}
    recipies = initialize_recipes()


    #remove all duplicates from the list, alternatively store duplicates in a separate list
    detected_ingredients = list(dict.fromkeys(detected_ingredients))
    #convert the list of codes to list of ingredients
    for i in range(len(detected_ingredients)):
        detected_ingredients[i] = ingredients_codes[detected_ingredients[i]]

    print("Detected ingredients: ",detected_ingredients)
            
    return detected_ingredients






def suggest_recipe(detected_ingredients, recipes):
    detected_recipies = []
    #check if the detected ingredients match any of the recipies
    for recipe in recipes:
        ingredients = recipe["ingredients"]

        if(set(ingredients).issubset(set(detected_ingredients))):
            print("Recipe found: ", recipe["name"])
            detected_recipies.append(recipe)
    return detected_recipies



def main():
    take_picture()
    detected_ingredients = detect_ingredients()
    recipes = initialize_recipes()
    suggested_recipes = suggest_recipe(detected_ingredients, recipes)

    print("\n \nBased on your detected ingredients, we suggest the following recipes: \n")
    for recipe in suggested_recipes:
        print("\n",recipe["name"])

        if "URL" in recipe:
            print("URL: ", recipe["URL"])

main()
