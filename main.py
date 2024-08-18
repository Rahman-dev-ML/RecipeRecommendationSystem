from flask import Flask, render_template, request, jsonify
from flask_pymongo import PyMongo
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import ast
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=30000)
db = client.test
app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/RecipeDB"
mongo = PyMongo(app)


# Load the dataset
df = pd.read_csv(r'C:\Users\Dell\Desktop\RAW_recipes-f.csv')
df.dropna(subset=['ingredients'], inplace=True)
df['ingredients'] = df['ingredients'].astype(str)
df['cleaned_ingredients'] = df['ingredients'].apply(lambda x: ', '.join(x.split(',')))

vectorizer = CountVectorizer()
ingredient_matrix = vectorizer.fit_transform(df['cleaned_ingredients'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/by_ingredient', methods=['GET', 'POST'])
def by_ingredient():
    if request.method == 'POST':
        user_ingredients = request.form['ingredients']
        user_vector = vectorizer.transform([user_ingredients.lower()])
        similarity_scores = cosine_similarity(user_vector, ingredient_matrix)
        recommended_indices = similarity_scores.argsort()[0][-5:][::-1]
        recommended_recipes = df['name'].iloc[recommended_indices].tolist()

        # Storing the recommended recipes into the Recipes collection
        mongo.db.Recipes.insert_one({
            "searched_ingredients": user_ingredients.split(", "),
            "recommended_recipes": recommended_recipes
        })

        return render_template('by_ingredient.html', recipes=recommended_recipes, ingredients=user_ingredients)

    return render_template('by_ingredient.html')

@app.route('/by_diet')
def by_diet():
    # Fetch unique diet types from the Recommendations collection
    diet_types = mongo.db.Recommendations.distinct("Diet_type")
    return render_template('by_diet.html', diets=diet_types)

@app.route('/recipes/<diet_type>')
def recipes(diet_type):
    # Fetch all recipes for the specified diet type from the Recommendations collection
    recipes = mongo.db.Recommendations.find({"Diet_type": diet_type})
    return render_template('recipes.html', recipes=list(recipes), diet_type=diet_type)
@app.route('/find_recipes', methods=['POST'])
def find_recipes():
    query = request.form['recipe_name'].lower()
    results = mongo.db.ff.find({"name": {"$regex": query, "$options": "i"}})
    recipes = []
    for recipe in results:
        recipe['ingredients'] = ast.literal_eval(recipe['ingredients'])  # Convert string list to actual list
        recipe['steps'] = ast.literal_eval(recipe['steps'])  # Convert string list to actual list
        recipes.append(recipe)
    return render_template('recipe_detail.html', recipes=recipes, search_query=query)
@app.route('/search_recipes', methods=['POST'])
def search_recipes():
    recipe_name = request.form['recipe_name'].lower()
    recipes = mongo.db.Recommendations.find({"Recipe_name": {"$regex": recipe_name, "$options": "i"}})
    return render_template('recipes.html', recipes=list(recipes), search_query=recipe_name)
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/by_method')
def by_method():
    # Placeholder for method-based recommendation logic
    return render_template('by_method.html')
@app.route('/pick_of_week')
def pick_of_the_week():
    return render_template('pick_of_week.html')

if __name__ == '__main__':
    app.run(debug=True)
