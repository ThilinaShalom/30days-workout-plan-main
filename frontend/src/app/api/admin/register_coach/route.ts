import { NextRequest, NextResponse } from 'next/server';
import { adminAuth, adminDb } from '@/lib/firebase-admin';
import { getSession } from '@/lib/session';

export async function POST(request: NextRequest) {
    try {
        const session = await getSession();
        if (!session || session.user_type !== 'admin') {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }

        const data = await request.json();
        const { email, password, coach_name, specialization, profile_pic_url, services } = data;

        if (!email || !password || !coach_name) {
            return NextResponse.json(
                { error: 'Email, password, and coach name are required' },
                { status: 400 }
            );
        }

        // Create coach in Firebase Auth
        const userRecord = await adminAuth().createUser({
            email,
            password,
            displayName: coach_name,
        });

        // Store coach data in Firestore
        await adminDb().collection('coaches').doc(userRecord.uid).set({
            user_name: coach_name,
            username: coach_name,
            email,
            user_type: 'coach',
            specialization: specialization || '',
            profile_pic_url: profile_pic_url || '',
            services: Array.isArray(services) ? services : [],
            created_at: new Date().toISOString(),
        });

        return NextResponse.json({
            success: true,
            message: 'Coach registered successfully',
            coach_id: userRecord.uid,
        });
    } catch (error: any) {
        console.error('Error registering coach:', error);
        return NextResponse.json(
            { error: `Failed to register coach: ${error.message}` },
            { status: 500 }
        );
    }
}
