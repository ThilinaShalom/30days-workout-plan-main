from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import firebase_admin
from firebase_admin import credentials, firestore, auth
from google.api_core.exceptions import DeadlineExceeded
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from preprocess import extract_numeric

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, template_folder='templates')
app.config['JSON_AS_ASCII'] = False
app.secret_key = 'your_secret_key'  # Replace with a real secret key

# Initialize Firebase
cred = credentials.Certificate('hdproject-6e51c-firebase-adminsdk-4e5te-d7102a3fe3.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Add dataset loading
bmi_df = pd.read_csv('data/bmi.csv')
meals_df = pd.read_csv('data/mealplans.csv')
nutrition_df = pd.read_csv('data/nutrition.csv')

# Clean and preprocess the BMI data
bmi_df.dropna(inplace=True)
bmi_df['Bmi'] = bmi_df['Weight'] / (bmi_df['Height'] ** 2)
bmi_df['BmiClass'] = pd.cut(bmi_df['Bmi'], bins=[0, 18.5, 24.9, 29.9, 34.9, 39.9, np.inf],
                           labels=['Underweight', 'Normal weight', 'Overweight', 'Obese Class 1', 'Obese Class 2', 'Obese Class 3'])

# Normalize the nutritional data
columns_to_normalize = ['calories', 'total_fat', 'cholesterol', 'sodium', 'fiber', 'protein']

for col in columns_to_normalize:
    if col not in nutrition_df.columns:
        print(f"Column {col} not found in the dataset")
    else:
        nutrition_df[col] = nutrition_df[col].apply(extract_numeric)

scaler = StandardScaler()
nutrition_df[columns_to_normalize] = scaler.fit_transform(nutrition_df[columns_to_normalize])

# Load the trained model
model = joblib.load('models/model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

def generate_workout_plan(user_info):
    # Convert user info to format expected by model
    user_data = pd.DataFrame([{
        'question3': float(user_info.get('weight_in_kg', 0)),
        'question4': float(user_info.get('height_in_cm', 0)),
        'question5': float(user_info.get('age', 0)),
        'question6': user_info.get('equipment', ''),
        'question7': float(user_info.get('days_per_week', 0)),
        'question11': float(user_info.get('Sleeping_Hours_per_day', 0))
    }])
    
    # Prepare features for model
    user_features = user_data[['question3', 'question4', 'question5', 'question7', 'question11']].astype(float)
    user_features['dummy_feature'] = 0
    
    # Scale features
    user_scaler = StandardScaler()
    user_features = user_scaler.fit_transform(user_features)
    
    # Get cluster prediction
    cluster = model.predict(user_features)[0]
    
    # Generate plan based on cluster
    if cluster == 0:
        focus = "Cardio-focused"
    elif cluster == 1:
        focus = "Strength training-focused"
    else:
        focus = "Flexibility and balance-focused"
        
    # Generate the workout plan
    plan = {}
    for day in range(1, 31):
        if focus == "Cardio-focused":
            if day % 7 in [1, 4]:
                plan[day] = f"Cardio - {user_data['question6'][0]} for {user_data['question7'][0]} minutes"
            elif day % 7 in [2, 5]:
                plan[day] = "Strength Training"
            elif day % 7 == 3:
                plan[day] = "Rest"
            else:
                plan[day] = "Flexibility Exercises"
        elif focus == "Strength training-focused":
            if day % 7 in [1, 4]:
                plan[day] = "Strength Training"
            elif day % 7 in [2, 5]:
                plan[day] = f"Cardio - {user_data['question6'][0]} for {user_data['question7'][0]} minutes"
            elif day % 7 == 3:
                plan[day] = "Rest"
            else:
                plan[day] = "Flexibility Exercises"
        else:
            if day % 7 in [1, 4]:
                plan[day] = "Flexibility Exercises"
            elif day % 7 in [2, 5]:
                plan[day] = f"Cardio - {user_data['question6'][0]} for {user_data['question7'][0]} minutes"
            elif day % 7 == 3:
                plan[day] = "Rest"
            else:
                plan[day] = "Strength Training"
                
    # Convert plan to string format
    plan_str = "30-Day Workout Plan:\n\n"
    for day, workout in plan.items():
        plan_str += f"Day {day}: {workout}\n"
        
    # Add meal plan
    meal_plan = generate_meal_plan(focus if focus == "High-protein" else "Balanced", user_data)
    plan_str += "\nMeal Plan:\n"
    for day, meals in meal_plan.items():
        plan_str += f"\nDay {day}:\n"
        for meal, food in meals.items():
            plan_str += f"{meal.title()}: {food}\n"
            
    return plan_str

def generate_meal_plan(focus, user_data):
    plan = {}
    for day in range(1, 31):
        if focus == "High-protein":
            plan[day] = {
                "breakfast": "Eggs",
                "lunch": "Chicken Salad",
                "dinner": "Grilled Fish"
            }
        elif focus == "Balanced":
            plan[day] = {
                "breakfast": "Oatmeal",
                "lunch": "Turkey Sandwich",
                "dinner": "Stir-fried Vegetables"
            }
        else:
            plan[day] = {
                "breakfast": "Smoothie",
                "lunch": "Salad",
                "dinner": "Grilled Chicken"
            }
    return plan

# Admin Registration
@app.route('/admin/register', methods=['GET', 'POST'])
def admin_register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Hash password before storing
        hashed_password = generate_password_hash(password)

        # Create the admin user in Firestore
        db.collection('admins').document(username).set({
            'username': username,
            'password': hashed_password
        })
        return redirect(url_for('admin_login'))
    
    return render_template('admin/admin_registration.html')

# Admin Login
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Retrieve the admin from Firestore
        admin_doc = db.collection('admins').document(username).get()
        if admin_doc.exists:
            admin_data = admin_doc.to_dict()
            if check_password_hash(admin_data['password'], password):
                session['is_admin'] = True
                session['admin_username'] = username
                return redirect(url_for('admin_dashboard'))
        
        return 'Login failed. Check your username and password.', 401
    
    return render_template('admin/admin_login.html')

# Admin Dashboard
@app.route('/admin/dashboard')
def admin_dashboard():
    if 'is_admin' not in session:
        return redirect(url_for('admin_login'))

    # Fetch all coaches from Firestore
    user_data = db.collection('users').where('user_type', '==', 'coach')
    coaches = [doc.to_dict() for doc in user_data.stream()]

    return render_template('admin/admin_dashboard.html', coaches=coaches)

# Coach Registration by Admin
@app.route('/admin/register_coach', methods=['POST'])
def register_coach():
    if 'is_admin' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    user_type = request.form['user_type']
    coach_name = request.form['coach_name']
    email = request.form['email']
    password = request.form['password']
    specialization = request.form['specialization']
    profile_pic_url = request.form['profile_pic_url']
    services = request.form.getlist('services')  # Assuming services is a list

    # Hash password before storing
    hashed_password = generate_password_hash(password)

    # Store the new coach in Firestore
    user = auth.create_user(email=email, password=password)
    db.collection('users').document(user.uid).set({
        'user_type': user_type,
        'username': coach_name,
        'email': email,
        'password': hashed_password,
        'specialization': specialization,
        'profile_pic_url': profile_pic_url,
        'services': services
    })

    return jsonify({'success': True}), 200

@app.route('/admin/delete_coach/<coach_email>', methods=['POST'])
def delete_coach(coach_email):
    if 'is_admin' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        # Find the coach document by email and delete it
        coach_query = db.collection('users').where('email', '==', coach_email).limit(1)
        coach_docs = coach_query.get()
        
        if not coach_docs:
            return jsonify({'error': 'Coach not found'}), 404
        
        for doc in coach_docs:
            doc.reference.delete()
        
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/get_coach/<coach_email>')
def get_coach(coach_email):
    if 'is_admin' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    coach_query = db.collection('users').where('email', '==', coach_email).limit(1)
    coach_docs = coach_query.get()
    
    if not coach_docs:
        return jsonify({'error': 'Coach not found'}), 404
    
    coach_data = coach_docs[0].to_dict()
    return jsonify(coach_data)

@app.route('/admin/edit_coach/<coach_email>', methods=['POST'])
def edit_coach(coach_email):
    if 'is_admin' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    coach_query = db.collection('users').where('email', '==', coach_email).limit(1)
    coach_docs = coach_query.get()
    
    if not coach_docs:
        return jsonify({'error': 'Coach not found'}), 404
    
    coach_doc = coach_docs[0]
    
    updated_data = {
        'username': request.form['coach_name'],
        'specialization': request.form['specialization'],
        'profile_pic_url': request.form['profile_pic_url'],
        'services': request.form.getlist('services')
    }
    
    new_email = request.form['email']
    
    # Update email in Firebase Auth if it has changed
    if new_email != coach_email:
        try:
            user = auth.get_user_by_email(coach_email)
            auth.update_user(user.uid, email=new_email)
            updated_data['email'] = new_email
        except Exception as e:
            return jsonify({'error': f'Failed to update email: {str(e)}'}), 400
    
    coach_doc.reference.update(updated_data)
    return jsonify({'success': True})

@app.route('/admin/reset_coach_password/<coach_email>', methods=['POST'])
def reset_coach_password(coach_email):
    if 'is_admin' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        auth.generate_password_reset_link(coach_email)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': f'Failed to send password reset email: {str(e)}'}), 400

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user_name = request.form['user_name']
        user_type = request.form['user_type']
        try:
            user = auth.create_user(email=email, password=password)
            db.collection('users').document(user.uid).set({
                'user_name': user_name,
                'email': email,
                'user_type': user_type
            })
            return redirect(url_for('login'))
        except Exception as e:
            return f'Registration failed: {str(e)}'
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            user = auth.get_user_by_email(email)
            user_data = db.collection('users').document(user.uid).get().to_dict()
            session['user_id'] = user.uid
            session['user_type'] = user_data['user_type']
            if user_data['user_type'] == 'customer':
                return redirect(url_for('customer_dashboard'))
            else:
                return redirect(url_for('coach_dashboard'))
        except Exception as e:
            return f'Login failed: {str(e)}'
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/customer_dashboard')
def customer_dashboard():
    if 'user_id' not in session or session['user_type'] != 'customer':
        return redirect(url_for('login'))

    # Fetch user data
    user_data = db.collection('users').document(session['user_id']).get().to_dict()
    user_name = user_data.get('user_name', 'Customer')  # Use 'Customer' as fallback if name is not set

    # Fetch plans for the customer
    plans_ref = db.collection('plans').where('user_id', '==', session['user_id'])
    plans = []
    for doc in plans_ref.stream():
        plan = doc.to_dict()
        plan['id'] = doc.id  # Add the document ID to the plan data
        plans.append(plan)

    return render_template('customer_dashboard.html', user_name=user_name, plans=plans)

@app.route('/coach_dashboard')
def coach_dashboard():
    if 'user_id' not in session or session['user_type'] != 'coach':
        return redirect(url_for('login'))

    # Fetch plans that have been sent for review
    plans_ref = db.collection('plans').where('status', '==', 'requested')
    plans = []
    for doc in plans_ref.stream():
        plan = doc.to_dict()
        plan['id'] = doc.id
        plans.append(plan)

    return render_template('coach_dashboard.html', plans=plans)

@app.route('/tell_coach/<plan_id>', methods=['POST'])
def tell_coach(plan_id):
    if 'user_id' not in session or session['user_type'] != 'customer':
        return jsonify({'error': 'Unauthorized'}), 401

    # Update the plan status to "requested" and mark it as sent to the coach
    try:
        plan_ref = db.collection('plans').document(plan_id)
        plan_doc = plan_ref.get()
        
        if not plan_doc.exists:
            return jsonify({'error': 'Plan not found'}), 404
        
        plan_data = plan_doc.to_dict()
        if plan_data['user_id'] != session['user_id']:
            return jsonify({'error': 'Unauthorized'}), 401
        
        plan_ref.update({
            'status': 'requested'
        })
        return jsonify({'success': True}), 200
    except Exception as e:
        print(f"Error in tell_coach: {str(e)}")  # Log the error
        return jsonify({'error': str(e)}), 500

@app.route('/review_plan/<plan_id>', methods=['POST'])
def review_plan(plan_id):
    if 'user_id' not in session or session['user_type'] != 'coach':
        return jsonify({'error': 'Unauthorized'}), 401
    
    coach_comment = request.form.get('coach_comment')
    action = request.form.get('action')

    print(f"Received data: plan_id={plan_id}, coach_comment={coach_comment}, action={action}")  # Debug log

    if not coach_comment or not action:
        print(f"Missing fields: coach_comment={coach_comment}, action={action}")  # Debug log
        return jsonify({'error': 'Missing required fields'}), 400

    plan_ref = db.collection('plans').document(plan_id)
    
    try:
        plan_doc = plan_ref.get()
        if not plan_doc.exists:
            return jsonify({'error': 'Plan not found'}), 404

        new_status = 'approved' if action == 'approve' else 'rejected'
        
        update_data = {
            'coach_comment': coach_comment,
            'status': new_status
        }
        print(f"Updating plan {plan_id} with data: {update_data}")  # Debug log
        plan_ref.update(update_data)
        
        print(f"Plan {plan_id} {new_status} successfully")
        return jsonify({'success': True, 'status': new_status}), 200
    except Exception as e:
        print(f"Error in review_plan: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/delete_plan/<plan_id>', methods=['POST'])
def delete_plan(plan_id):
    if 'user_id' not in session or session['user_type'] != 'customer':
        return jsonify({'error': 'Unauthorized'}), 401

    plan_ref = db.collection('plans').document(plan_id)
    
    try:
        plan_doc = plan_ref.get()
        if not plan_doc.exists:
            return jsonify({'error': 'Plan not found'}), 404
        
        plan_data = plan_doc.to_dict()
        if plan_data['user_id'] != session['user_id']:
            return jsonify({'error': 'Unauthorized'}), 401
        
        plan_ref.delete()
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if 'user_id' not in session or session['user_type'] != 'customer':
        return redirect(url_for('login'))

    if request.method == 'POST':
        user_info = {
            'age': request.form.get('age'),
            'sex': request.form.get('sex'),
            'weight_in_kg': request.form.get('weight_in_kg'),
            'height_in_cm': request.form.get('height_in_cm'),
            'fitness_goal': request.form.get('fitness_goal'),
            'fitness_level': request.form.get('fitness_level'),
            'Cholesterol_level': request.form.get('Cholesterol_level'),
            'Pressure_level': request.form.get('Pressure_level'),
            'Sugar_level': request.form.get('Sugar_level'),
            'equipment': request.form.get('equipment'),
            'days_per_week': request.form.get('days_per_week'),
            'Sleeping_Hours_per_day': request.form.get('Sleeping_Hours_per_day'),
            'diet_type': request.form.get('diet_type'),
            'allergies': request.form.get('allergies'),
            'meals_per_day': request.form.get('meals_per_day'),
        }
        workout_plan = generate_workout_plan(user_info)
        
        try:
            # Save the generated plan to Firestore without sending it to the coach
            db.collection('plans').add({
                'user_id': session['user_id'],
                'plan': workout_plan,
                'fitness_goal': user_info['fitness_goal'],
                'status': 'not_sent',
                'user_info': user_info
            }, timeout=30)
        except DeadlineExceeded:
            return "Firestore request timed out. Please try again later.", 504
        
        return render_template('result.html', workout_plan=workout_plan)
    return render_template('generate_form.html')

if __name__ == '__main__':
    app.run(debug=True)
