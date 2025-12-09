export interface WorkoutExercise {
    name: string;
    desc: string;
    equipment: string;
    sets: number;
    reps: number;
    rating: number;
    intensity: string | number;
}

export interface DayWorkout {
    type: string;
    exercises: WorkoutExercise[];
    intensity: string | number;
    notes: string;
}

export interface UserInfo {
    equipment: string;
    fitness_level: number | string;
    days_per_week: number;
    intensity?: number | string;
    [key: string]: any;
}

// Equipment mapping
const EQUIPMENT_MAP: { [key: string]: string } = {
    'none': 'Body Only',
    'bands': 'Bands',
    'barbell': 'Barbell',
    'dumbbell': 'Dumbbell',
    'cable': 'Cable',
    'machine': 'Machine',
    'kettlebell': 'Kettlebells',
    'medicine ball': 'Medicine Ball',
    'exercise ball': 'Exercise Ball',
};

// Fitness level mapping
const LEVEL_MAP: { [key: number]: string } = {
    1: 'Beginner',
    2: 'Intermediate',
    3: 'Expert',
};

// Load workout data (will be replaced with actual JSON data)
let workoutsData: any[] = [];

export function initializeWorkoutsData(data: any[]) {
    workoutsData = data;
}

export function generateWorkoutPlan(userInfo: UserInfo): { [key: string]: DayWorkout } {
    try {
        const equipment = EQUIPMENT_MAP[userInfo.equipment.toLowerCase()] || 'Body Only';
        const level = LEVEL_MAP[parseInt(userInfo.fitness_level.toString())] || 'Intermediate';
        const daysPerWeek = parseInt(userInfo.days_per_week.toString()) || 5;
        const fitnessLevel = parseInt(userInfo.fitness_level.toString()) || 2;

        // Filter workouts by equipment and level
        const filteredWorkouts = workoutsData.filter(
            (workout) =>
                workout.Equipment?.includes(equipment) &&
                workout.Level === level
        );

        // Group workouts by type
        const workoutGroups = {
            Cardio: filteredWorkouts
                .filter((w) => w.Type === 'Cardio')
                .sort((a, b) => (b.Rating || 0) - (a.Rating || 0)),
            Strength: filteredWorkouts
                .filter((w) => w.Type === 'Strength')
                .sort((a, b) => (b.Rating || 0) - (a.Rating || 0)),
            Flexibility: filteredWorkouts
                .filter((w) => ['Stretching', 'Plyometrics'].includes(w.Type))
                .sort((a, b) => (b.Rating || 0) - (a.Rating || 0)),
        };

        const totalDays = 30;
        const workoutDaysPerWeek = Math.min(daysPerWeek, 7);
        let totalWorkoutDays = (totalDays / 7) * workoutDaysPerWeek;
        let totalRestDays = totalDays - totalWorkoutDays;

        // Adjust rest days based on fitness level
        if (fitnessLevel === 1) {
            totalRestDays = Math.min(totalRestDays + 2, totalDays - 5);
        } else if (fitnessLevel === 3) {
            totalRestDays = Math.max(totalRestDays - 2, 2);
        }

        const plan: { [key: string]: DayWorkout } = {};

        // Randomly select rest days
        const restDayIndices = new Set<number>();
        while (restDayIndices.size < totalRestDays) {
            restDayIndices.add(Math.floor(Math.random() * 30) + 1);
        }

        // Generate plan for each day
        for (let day = 1; day <= 30; day++) {
            if (restDayIndices.has(day)) {
                plan[day.toString()] = {
                    type: 'Rest',
                    exercises: [],
                    intensity: 'low',
                    notes: 'Focus on recovery',
                };
            } else {
                // Randomly select workout type with weighted probability
                const workoutTypes = ['Cardio', 'Strength', 'Flexibility'];
                const probabilities = [0.4, 0.4, 0.2];
                const random = Math.random();
                let cumulative = 0;
                let workoutType = 'Cardio';

                for (let i = 0; i < workoutTypes.length; i++) {
                    cumulative += probabilities[i];
                    if (random < cumulative) {
                        workoutType = workoutTypes[i];
                        break;
                    }
                }

                const availableExercises = workoutGroups[workoutType as keyof typeof workoutGroups];

                let exercises: WorkoutExercise[] = [];
                if (availableExercises.length > 0) {
                    // Select top 3 exercises
                    const selectedExercises = availableExercises.slice(0, 3);
                    exercises = selectedExercises.map((ex) => ({
                        name: ex.Title || '',
                        desc: ex.Desc || '',
                        equipment: ex.Equipment || equipment,
                        sets: 3,
                        reps: workoutType === 'Strength' ? 12 : 30,
                        rating: parseFloat(ex.Rating) || 0,
                        intensity: userInfo.intensity || 'moderate',
                    }));
                } else {
                    // Fallback exercise
                    exercises = [
                        {
                            name: `Basic ${workoutType}`,
                            desc: 'Bodyweight exercise',
                            equipment: 'Body Only',
                            sets: 3,
                            reps: 12,
                            rating: 0,
                            intensity: userInfo.intensity || 'moderate',
                        },
                    ];
                }

                plan[day.toString()] = {
                    type: workoutType,
                    exercises,
                    intensity: userInfo.intensity || 'moderate',
                    notes: 'Focus on form',
                };
            }
        }

        return plan;
    } catch (error) {
        console.error('Error generating workout plan:', error);
        throw error;
    }
}
