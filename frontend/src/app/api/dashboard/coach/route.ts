import { NextRequest, NextResponse } from 'next/server';
import { adminDb } from '@/lib/firebase-admin';
import { getSession } from '@/lib/session';

export async function GET(request: NextRequest) {
    try {
        console.log('Coach dashboard: Getting session...');
        const session = await getSession();
        console.log('Coach dashboard: Session:', session);

        if (!session) {
            console.log('Coach dashboard: No session found');
            return NextResponse.json({ error: 'No session' }, { status: 401 });
        }

        if (session.user_type !== 'coach') {
            console.log('Coach dashboard: User is not a coach:', session.user_type);
            return NextResponse.json({ error: 'Not a coach' }, { status: 401 });
        }

        console.log('Coach dashboard: Fetching plans...');
        // Get plans with "requested" status
        const plansSnapshot = await adminDb()
            .collection('plans')
            .where('status', '==', 'requested')
            .get();

        const plans = [];
        const fitnessGoals: { [key: string]: string } = {
            '0': 'Weight Loss',
            '1': 'Muscle Gain',
            '2': 'Endurance',
            '3': 'General Fitness',
        };

        for (const doc of plansSnapshot.docs) {
            const planData = doc.data();

            // Get user name
            const userDoc = await adminDb().collection('users').doc(planData.user_id).get();
            const userName = userDoc.exists ? userDoc.data()?.user_name || 'Unknown' : 'Unknown';

            plans.push({
                id: doc.id,
                ...planData,
                user_name: userName,
                fitness_goal: fitnessGoals[planData.user_data?.exercise_type?.toString() || ''] || 'Not specified',
            });
        }

        console.log('Coach dashboard: Found', plans.length, 'requested plans');
        return NextResponse.json({ plans });
    } catch (error: any) {
        console.error('Error fetching coach dashboard:', error);
        return NextResponse.json({ error: error.message }, { status: 500 });
    }
}
