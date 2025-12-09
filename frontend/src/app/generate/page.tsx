'use client';
import { useForm } from 'react-hook-form';
import { useRouter } from 'next/navigation';
import AppNavbar from '@/components/AppNavbar';
import api from '@/lib/api';
import { useState } from 'react';

export default function GeneratePlan() {
    const { register, handleSubmit, reset, formState: { errors } } = useForm();
    const router = useRouter();
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const onSubmit = async (data: any) => {
        setLoading(true);
        setError('');
        try {
            console.log('Submitting form data:', data);
            const response = await api.post('/generate', data);
            console.log('Response:', response.data);
            // Backend returns the complete plan data, not { success: true }
            if (response.data && (response.data.workout_plan || response.data.plan_id)) {
                alert('✅ Plan generated successfully!');
                router.push('/dashboard');
            } else {
                throw new Error('Invalid response from server');
            }
        } catch (error: any) {
            console.error('Error generating plan', error);
            const errorMsg = error.response?.data?.error || 'Failed to generate plan. Please try again.';
            setError(errorMsg);
            alert('❌ ' + errorMsg);
        } finally {
            setLoading(false);
        }
    };

    return (
        <>
            <AppNavbar brand="Generate Plan" />
            <div className="container mt-5 mb-5">
                <h2 className="text-center mb-4">Generate Your Personalized Fitness Plan</h2>
                {error && <div className="alert alert-danger">{error}</div>}
                <div className="card p-4" style={{ background: 'rgba(255, 255, 255, 0.05)', border: '1px solid rgba(0, 204, 255, 0.2)' }}>
                    <form onSubmit={handleSubmit(onSubmit)}>
                        {/* Personal Information */}
                        <h4 className="mb-3" style={{ color: '#00ccff' }}>Personal Information</h4>
                        <div className="mb-3">
                            <label className="form-label">Age</label>
                            <input type="number" className="form-control" {...register('age', { required: true, min: 16, max: 80 })} />
                        </div>
                        <div className="mb-3">
                            <label className="form-label">Biological Sex</label>
                            <select className="form-select" {...register('sex', { required: true })}>
                                <option value="male">Male</option>
                                <option value="female">Female</option>
                            </select>
                        </div>
                        <div className="mb-3">
                            <label className="form-label">Weight (kg)</label>
                            <input type="number" step="0.1" className="form-control" {...register('weight_in_kg', { required: true })} />
                        </div>
                        <div className="mb-3">
                            <label className="form-label">Height (cm)</label>
                            <input type="number" className="form-control" {...register('height_in_cm', { required: true })} />
                        </div>

                        {/* Fitness Profile */}
                        <h4 className="mb-3 mt-4" style={{ color: '#00ccff' }}>Fitness Profile</h4>
                        <div className="mb-3">
                            <label className="form-label">Primary Fitness Goal</label>
                            <select className="form-select" {...register('fitness_goal', { required: true })}>
                                <option value="weight_loss">Weight Loss</option>
                                <option value="muscle_gain">Muscle Gain</option>
                                <option value="endurance">Endurance</option>
                                <option value="strength">Strength</option>
                                <option value="flexibility">Flexibility</option>
                                <option value="general_fitness">General Fitness</option>
                            </select>
                        </div>
                        <div className="mb-3">
                            <label className="form-label">Current Fitness Level</label>
                            <select className="form-select" {...register('fitness_level', { required: true })}>
                                <option value="1">Beginner (No regular exercise)</option>
                                <option value="2">Novice (Occasional exercise)</option>
                                <option value="3">Intermediate (Regular exercise)</option>
                                <option value="4">Advanced (Consistent training)</option>
                                <option value="5">Expert (Athletic background)</option>
                            </select>
                        </div>

                        {/* Health Information */}
                        <h4 className="mb-3 mt-4" style={{ color: '#00ccff' }}>Health Information</h4>
                        <div className="mb-3">
                            <label className="form-label">Blood Pressure</label>
                            <select className="form-select" {...register('blood_pressure', { required: true })}>
                                <option value="normal">Normal (120/80 or lower)</option>
                                <option value="elevated">Elevated (120-129/80)</option>
                                <option value="high">High (130+/80+)</option>
                                <option value="unknown">Unknown</option>
                            </select>
                        </div>

                        {/* Exercise Preferences */}
                        <h4 className="mb-3 mt-4" style={{ color: '#00ccff' }}>Exercise Preferences</h4>
                        <div className="mb-3">
                            <label className="form-label">Available Equipment</label>
                            <select className="form-select" {...register('equipment', { required: true })}>
                                <option value="none">No Equipment (Bodyweight Only)</option>
                                <option value="dumbbell">Dumbbells</option>
                                <option value="bands">Resistance Bands</option>
                                <option value="barbell">Barbell Set</option>
                                <option value="machine">Gym Machines</option>
                                <option value="cable">Cable/Cardio Equipment</option>
                            </select>
                        </div>
                        <div className="mb-3">
                            <label className="form-label">Preferred Training Days per Week</label>
                            <input type="number" className="form-control" {...register('days_per_week', { required: true, min: 2, max: 6 })} />
                        </div>
                        <div className="mb-3">
                            <label className="form-label">Preferred Exercise Intensity</label>
                            <select className="form-select" {...register('intensity', { required: true })}>
                                <option value="-1">Low (Light exercise)</option>
                                <option value="0">Moderate (Some effort)</option>
                                <option value="1">High (Heavy breathing)</option>
                            </select>
                        </div>
                        <div className="mb-3">
                            <label className="form-label">Preferred Exercise Types</label>
                            <select className="form-select" {...register('exercise_type', { required: true })}>
                                <option value="0">Cardiovascular</option>
                                <option value="1">Strength Training</option>
                                <option value="2">High-Intensity Interval</option>
                                <option value="3">Flexibility/Stretching</option>
                            </select>
                        </div>

                        {/* Lifestyle */}
                        <h4 className="mb-3 mt-4" style={{ color: '#00ccff' }}>Lifestyle & Recovery</h4>
                        <div className="mb-3">
                            <label className="form-label">Average Sleep Hours</label>
                            <input type="number" step="0.5" className="form-control" {...register('sleep_hours', { required: true, min: 4, max: 12 })} />
                        </div>
                        <div className="mb-3">
                            <label className="form-label">Stress Level</label>
                            <select className="form-select" {...register('stress_level', { required: true })}>
                                <option value="1">Low</option>
                                <option value="2">Moderate</option>
                                <option value="3">High</option>
                            </select>
                        </div>
                        <div className="mb-3">
                            <label className="form-label">Daily Activity Level</label>
                            <select className="form-select" {...register('activity_level', { required: true })}>
                                <option value="-1">Sedentary</option>
                                <option value="0">Moderate</option>
                                <option value="1">Active</option>
                            </select>
                        </div>

                        {/* Nutrition */}
                        <h4 className="mb-3 mt-4" style={{ color: '#00ccff' }}>Nutrition Preferences</h4>
                        <div className="mb-3">
                            <label className="form-label">Diet Preference</label>
                            <select className="form-select" {...register('diet_type', { required: true })}>
                                <option value="balanced">Balanced</option>
                                <option value="keto">Ketogenic</option>
                                <option value="vegetarian">Vegetarian</option>
                                <option value="vegan">Vegan</option>
                                <option value="mediterranean">Mediterranean</option>
                                <option value="paleo">Paleo</option>
                            </select>
                        </div>
                        <div className="mb-3">
                            <label className="form-label">Preferred Meals per Day</label>
                            <select className="form-select" {...register('meals_per_day', { required: true })}>
                                <option value="3">3 meals</option>
                                <option value="4">4 meals</option>
                                <option value="5">5 meals</option>
                                <option value="6">6 meals</option>
                            </select>
                        </div>
                        <div className="mb-3">
                            <label className="form-label">Food Allergies/Restrictions (Optional)</label>
                            <input type="text" className="form-control" {...register('allergies')} placeholder="E.g., dairy, nuts, gluten (leave blank if none)" />
                        </div>
                        <div className="mb-3">
                            <label className="form-label">Daily Calorie Target *</label>
                            <input type="number" className="form-control" {...register('calorie_target', { required: true, min: 1200, max: 4000 })} placeholder="e.g., 2000" />
                        </div>
                        <div className="mb-3">
                            <label className="form-label">Macro Nutrient Preference</label>
                            <select className="form-select" {...register('macro_preference', { required: true })}>
                                <option value="balanced">Balanced (30/40/30)</option>
                                <option value="high_protein">High Protein (40/40/20)</option>
                                <option value="low_carb">Low Carb (50/10/40)</option>
                                <option value="high_carb">High Carb (30/50/20)</option>
                            </select>
                        </div>
                        <div className="mb-3">
                            <label className="form-label">Dietary Restrictions (Optional)</label>
                            <div className="form-check">
                                <input className="form-check-input" type="checkbox" value="gluten_free" {...register('restrictions')} />
                                <label className="form-check-label">Gluten Free</label>
                            </div>
                            <div className="form-check">
                                <input className="form-check-input" type="checkbox" value="lactose_free" {...register('restrictions')} />
                                <label className="form-check-label">Lactose Free</label>
                            </div>
                            <div className="form-check">
                                <input className="form-check-input" type="checkbox" value="nut_free" {...register('restrictions')} />
                                <label className="form-check-label">Nut Free</label>
                            </div>
                            <small className="text-muted">Leave unchecked if no restrictions</small>
                        </div>

                        <div className="d-flex justify-content-end gap-3 mt-4">
                            <button type="button" className="btn btn-outline-secondary" onClick={() => router.back()}>Cancel</button>
                            <button type="button" className="btn btn-outline-warning" onClick={() => reset()}>Clear</button>
                            <button type="submit" className="btn btn-primary" disabled={loading}>
                                {loading ? 'Generating...' : 'Generate Personalized Plan'}
                            </button>
                        </div>
                    </form>
                </div>
            </div>
        </>
    );
}
