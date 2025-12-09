export interface NutritionPlan {
    daily_targets: {
        calories: number;
        protein: number;
        carbs: number;
        fat: number;
        fiber: number;
    };
    meals: {
        [mealName: string]: {
            calories: number;
            protein: number;
            carbs: number;
            fat: number;
            fiber: number;
        };
    };
    diet_type: string;
    macro_split: {
        protein: number;
        carbs: number;
        total_fat: number;
    };
}

export interface UserData {
    calories: number;
    diet_type: string;
    macro_preference: string;
    meals_per_day: number;
    [key: string]: any;
}

// Macro ratio definitions
const MACRO_RATIOS: {
    [key: string]: { protein: number; carbs: number; total_fat: number };
} = {
    balanced: { protein: 0.3, carbs: 0.4, total_fat: 0.3 },
    high_protein: { protein: 0.4, carbs: 0.4, total_fat: 0.2 },
    low_carb: { protein: 0.5, carbs: 0.1, total_fat: 0.4 },
    high_carb: { protein: 0.3, carbs: 0.5, total_fat: 0.2 },
};

export function generateNutritionPlan(userData: UserData): NutritionPlan {
    try {
        const dietType = userData.diet_type;
        const mealsPerDay = userData.meals_per_day;
        const calorieTarget = parseFloat(userData.calories.toString());
        const macroPref = userData.macro_preference;

        // Get macro ratios
        const macroRatios = MACRO_RATIOS[macroPref] || MACRO_RATIOS.balanced;

        // Calculate macronutrient targets
        // Protein and carbs: 4 kcal/g, Fat: 9 kcal/g
        const proteinTarget = (calorieTarget * macroRatios.protein) / 4;
        const carbTarget = (calorieTarget * macroRatios.carbs) / 4;
        const fatTarget = (calorieTarget * macroRatios.total_fat) / 9;
        const fiberTarget = calorieTarget * (dietType === 'high_carb' ? 0.016 : 0.014);

        // Define meal names based on meals per day
        let mealNames: string[];
        if (mealsPerDay <= 3) {
            mealNames = ['Breakfast', 'Lunch', 'Dinner'].slice(0, mealsPerDay);
        } else {
            mealNames = ['Breakfast', 'Snack 1', 'Lunch', 'Snack 2', 'Dinner'].slice(0, mealsPerDay);
        }

        // Distribute macros evenly across meals
        const meals: NutritionPlan['meals'] = {};
        mealNames.forEach((name) => {
            meals[name] = {
                calories: calorieTarget / mealsPerDay,
                protein: proteinTarget / mealsPerDay,
                carbs: carbTarget / mealsPerDay,
                fat: fatTarget / mealsPerDay,
                fiber: fiberTarget / mealsPerDay,
            };
        });

        // Compile nutrition plan
        const nutritionPlan: NutritionPlan = {
            daily_targets: {
                calories: calorieTarget,
                protein: proteinTarget,
                carbs: carbTarget,
                fat: fatTarget,
                fiber: fiberTarget,
            },
            meals,
            diet_type: dietType,
            macro_split: macroRatios,
        };

        return nutritionPlan;
    } catch (error) {
        console.error('Error generating nutrition plan:', error);
        throw error;
    }
}
