import { NextRequest, NextResponse } from 'next/server';
import { adminDb } from '@/lib/firebase-admin';
import { getSession } from '@/lib/session';
import { predictCluster, UserFeatures } from '@/lib/ml/predict';
import { generateWorkoutPlan, initializeWorkoutsData } from '@/lib/workout-generator';
import { generateNutritionPlan } from '@/lib/nutrition-generator';
import workoutsData from '@/data/workouts.json';

// Initialize workouts data
initializeWorkoutsData(workoutsData);

export async function POST(request: NextRequest) {
    try {
        // Check session
        const session = await getSession();
        if (!session) {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }

        const formData = await request.json();

        // Process form data
        const processedData = processFormData(formData);

        // Create features for ML prediction
        const features: UserFeatures = {
            weight: processedData.weight,
            height: processedData.height,
            age: processedData.age,
            bmi: processedData.bmi,
            days_per_week: processedData.days_per_week,
            sleep_hours: processedData.sleep_hours,
            calories: processedData.calories,
            protein: processedData.protein,
            carbohydrate: processedData.carbohydrate,
            total_fat: processedData.total_fat,
            fiber: processedData.fiber,
            intensity: processedData.intensity,
            exercise_type: processedData.exercise_type,
            rating: processedData.rating,
        };

        // Predict cluster using ML model
        const prediction = predictCluster(features);

        // Generate workout plan
        const workoutPlan = generateWorkoutPlan(processedData);

        // Generate nutrition plan
        const nutritionPlan = generateNutritionPlan(processedData);

        // Format complete plan
        const completePlan = {
            workout_plan: workoutPlan,
            nutrition_plan: nutritionPlan,
            overview: {
                total_days: 30,
                workout_days: Object.values(workoutPlan).filter((d) => d.type !== 'Rest').length,
                rest_days: Object.values(workoutPlan).filter((d) => d.type === 'Rest').length,
            },
        };

        // Save plan to Firestore
        const planData = {
            user_id: session.user_id,
            created_at: new Date().toISOString(),
            status: 'new',
            workout_plan: workoutPlan,
            nutrition_plan: nutritionPlan,
            user_data: processedData,
            cluster: prediction.cluster,
            cluster_info: prediction.clusterInfo,
            coach_comment: '',
            coach_id: null,
        };

        const planRef = await adminDb().collection('plans').add(planData);

        return NextResponse.json({
            ...completePlan,
            plan_id: planRef.id,
            cluster: prediction.cluster,
            cluster_info: prediction.clusterInfo,
        });
    } catch (error: any) {
        console.error('Error generating plan:', error);
        return NextResponse.json({ error: error.message }, { status: 400 });
    }
}

function processFormData(formData: any) {
    // Define required fields
    const requiredFields = [
        'weight_in_kg',
        'height_in_cm',
        'age',
        'days_per_week',
        'sleep_hours',
        'intensity',
        'exercise_type',
        'calorie_target',
        'macro_preference',
        'diet_type',
        'equipment',
        'fitness_level',
        'meals_per_day',
    ];

    const missingFields = requiredFields.filter((field) => !formData[field]);
    if (missingFields.length > 0) {
        throw new Error(`Missing fields: ${missingFields.join(', ')}`);
    }

    // Convert height and weight
    const heightM = parseFloat(formData.height_in_cm) / 100;
    const weightKg = parseFloat(formData.weight_in_kg);
    const bmi = weightKg / (heightM * heightM);

    // Define macro ratios
    const macroRatios: { [key: string]: { protein: number; carbs: number; total_fat: number } } = {
        balanced: { protein: 0.3, carbs: 0.4, total_fat: 0.3 },
        high_protein: { protein: 0.4, carbs: 0.4, total_fat: 0.2 },
        low_carb: { protein: 0.5, carbs: 0.1, total_fat: 0.4 },
        high_carb: { protein: 0.3, carbs: 0.5, total_fat: 0.2 },
    };

    const macroPref = macroRatios[formData.macro_preference];

    // Create processed data
    const processedData = {
        weight: weightKg,
        height: heightM,
        age: parseInt(formData.age),
        bmi,
        days_per_week: parseInt(formData.days_per_week),
        sleep_hours: parseFloat(formData.sleep_hours),
        intensity: parseInt(formData.intensity),
        exercise_type: parseInt(formData.exercise_type),
        calories: parseFloat(formData.calorie_target),
        protein: macroPref.protein,
        carbohydrate: macroPref.carbs,
        total_fat: macroPref.total_fat,
        fiber: parseFloat(formData.calorie_target) * (formData.diet_type === 'high_carb' ? 0.016 : 0.014),
        equipment: formData.equipment,
        fitness_level: formData.fitness_level,
        rating: 0,
        diet_type: formData.diet_type,
        macro_preference: formData.macro_preference,
        meals_per_day: parseInt(formData.meals_per_day),
    };

    return processedData;
}
