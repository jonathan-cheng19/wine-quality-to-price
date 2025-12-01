from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import random
from datetime import datetime

app = Flask(__name__)

# Load the model and encoders
with open('wine_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('dropdown_data.pkl', 'rb') as f:
    dropdown_data = pickle.load(f)

# Load model statistics for better handling of unknown values
try:
    with open('model_stats.pkl', 'rb') as f:
        model_stats = pickle.load(f)
except:
    model_stats = {
        'mean_price': 39.15,
        'median_price': 18.20,
        'price_by_grape': {},
        'price_by_country': {},
        'price_by_rating': {},
    }

def generate_fancy_wine_name(grape, country, age):
    """Generate a fancy wine name using predefined components"""
    
    prefixes = [
        "Château", "Domaine", "Tenuta", "Bodega", "Estate", "Vignoble",
        "Clos", "Mas", "Finca", "Quinta", "Villa", "Castello"
    ]
    
    middle_parts = [
        "du Soleil", "des Anges", "Mystique", "Royal", "Imperial",
        "Noble", "Magnifique", "Étoile", "Luna", "del Monte",
        "Celestial", "Seraphim", "Velvet", "Crimson", "Golden"
    ]
    
    suffixes = [
        "Reserve", "Grand Cru", "Reserva Especial", "Limited Edition",
        "Private Selection", "Vintner's Choice", "Heritage", "Legacy",
        "Premier", "Exceptional", "Prestige", "Excellence"
    ]
    
    descriptors = [
        "Velvet", "Silk", "Midnight", "Starlight", "Moonlight",
        "Shadow", "Twilight", "Dawn", "Ethereal", "Divine"
    ]
    
    # Create the name
    prefix = random.choice(prefixes)
    middle = random.choice(middle_parts)
    descriptor = random.choice(descriptors)
    suffix = random.choice(suffixes)
    
    name_format = random.choice([
        f"{prefix} {middle} {descriptor} {suffix}",
        f"{prefix} {middle} {grape} {suffix}",
        f"{descriptor} {middle} {grape}",
        f"{prefix} {middle} {suffix}"
    ])
    
    current_year = datetime.now().year
    vintage = current_year - age
    
    return f"{name_format} {vintage}"

def generate_wine_story(name, grape, country, region, age, rating, num_ratings, predicted_price):
    """Generate a humorous wine backstory"""
    
    stories = [
        {
            "title": "The Mystical Origins",
            "slides": [
                f"Legend has it that {name} was discovered by a wandering sommelier who got lost in the vineyards of {region}, {country}.",
                f"The {grape} grapes used in this wine were blessed by exactly {num_ratings} wine critics who couldn't agree on anything except that it deserved a {rating} rating.",
                f"After {age} years of aging in barrels carved from ancient oak trees (or maybe just regular oak trees, the monks weren't very specific), this wine has achieved peak pretentiousness.",
                f"Our advanced AI algorithms, trained on thousands of wines and too much coffee, have determined this magnificent bottle is worth ${predicted_price:.2f}.",
                "Why this exact price? Because the AI said so, and who are we to argue with artificial intelligence? It's artificial AND intelligent!"
            ]
        },
        {
            "title": "The Price is Right (Sort Of)",
            "slides": [
                f"{name} comes from the prestigious region of {region}, where {grape} grapes grow under the judgmental gaze of {num_ratings} wine experts.",
                f"At {age} years old, this wine has seen things. It's been in the cellar longer than most relationships last.",
                f"With a rating of {rating} stars, this wine is almost, but not quite, entirely unlike vinegar.",
                f"Our machine learning model (which definitely didn't just guess) has scientifically determined this wine costs ${predicted_price:.2f}.",
                "This price was calculated using complex mathematical formulas, several random number generators, and a dartboard. Mostly the dartboard."
            ]
        },
        {
            "title": "A Tale of Grapes and Glory",
            "slides": [
                f"In the mystical lands of {country}, within the hallowed vineyards of {region}, {grape} grapes whispered secrets to winemakers.",
                f"These whispers were rated {rating} out of 5 by {num_ratings} people who take wine way too seriously (but in a good way).",
                f"For {age} glorious years, this wine has been maturing, contemplating life, and slowly becoming more expensive.",
                f"Through the power of regression analysis and a sprinkle of machine learning magic, we've divined that {name} is worth ${predicted_price:.2f}.",
                "Could it be worth more? Less? Who knows! But our model has an R² score above 0.5, so it's at least better than a coin flip!"
            ]
        },
        {
            "title": "The Algorithmic Appraisal",
            "slides": [
                f"Behold {name}, a {grape} from {region}, {country}—crafted by winemakers who definitely know what they're doing.",
                f"{num_ratings} sophisticated palates have bestowed upon it a {rating}-star rating, which is basically wine royalty.",
                f"This bottle has been aging for {age} years, gaining wisdom, complexity, and an inflated sense of self-worth.",
                f"Our AI model—trained on actual data and powered by hope—predicts this wine should cost ${predicted_price:.2f}.",
                "Is this price scientifically accurate? Well, it's based on machine learning, so it's at least as accurate as your weather app!"
            ]
        }
    ]
    
    return random.choice(stories)

@app.route('/')
def index():
    return render_template('index.html', 
                         grapes=dropdown_data['grapes'],
                         countries=dropdown_data['countries'],
                         regions=dropdown_data['regions'],
                         wineries=dropdown_data['wineries'])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract input data
        age = int(data['age'])
        rating = float(data['rating'])
        num_ratings = int(data['num_ratings'])
        grape = data['grape']
        country = data['country']
        region = data['region']
        winery = data['winery']
        
        # Encode categorical features with fallback for unknown values
        # Use the median encoded value if category not found
        try:
            grape_encoded = label_encoders['Grape'].transform([grape])[0]
        except:
            # Use most common grape encoding (0) as fallback
            grape_encoded = 0
            
        try:
            country_encoded = label_encoders['Country'].transform([country])[0]
        except:
            country_encoded = 0
            
        try:
            region_encoded = label_encoders['Region'].transform([region])[0]
        except:
            region_encoded = 0
            
        try:
            winery_encoded = label_encoders['Winery'].transform([winery])[0]
        except:
            winery_encoded = 0
        
        # Create feature array matching training data format
        features = np.array([[rating, num_ratings, age, country_encoded, 
                            region_encoded, winery_encoded, grape_encoded]])
        
        # Make prediction
        predicted_price = model.predict(features)[0]
        
        # Apply some business logic to ensure reasonable predictions
        # Consider rating influence
        rating_multiplier = (rating / 4.0) ** 2  # Square for exponential effect
        
        # Adjust based on statistical data if available
        base_price = predicted_price
        if grape in model_stats['price_by_grape']:
            grape_avg = model_stats['price_by_grape'][grape]
            # Blend prediction with grape average (70% prediction, 30% average)
            predicted_price = 0.7 * predicted_price + 0.3 * grape_avg
        
        # Ensure minimum price and reasonable bounds
        predicted_price = max(5.0, min(predicted_price, 2000.0))
        
        # Generate wine name and story
        wine_name = generate_fancy_wine_name(grape, country, age)
        story = generate_wine_story(wine_name, grape, country, region, age, 
                                   rating, num_ratings, predicted_price)
        
        return jsonify({
            'success': True,
            'predicted_price': round(predicted_price, 2),
            'wine_name': wine_name,
            'story': story
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
