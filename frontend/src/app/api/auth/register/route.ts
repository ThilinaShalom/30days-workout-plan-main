import { NextRequest, NextResponse } from 'next/server';
import { adminAuth, adminDb } from '@/lib/firebase-admin';
import { cookies } from 'next/headers';

export async function POST(request: NextRequest) {
    try {
        const data = await request.json();
        const { email, password, user_name, user_type } = data;

        if (!email || !password || !user_name || !user_type) {
            return NextResponse.json(
                { error: 'Missing required fields' },
                { status: 400 }
            );
        }

        // Create user in Firebase Auth
        const userRecord = await adminAuth().createUser({
            email,
            password,
        });

        // Store user data in Firestore
        await adminDb().collection('users').doc(userRecord.uid).set({
            user_name,
            email,
            user_type,
            created_at: new Date().toISOString(),
        });

        return NextResponse.json({
            success: true,
            message: 'Registration successful',
        });
    } catch (error: any) {
        console.error('Registration error:', error);
        return NextResponse.json(
            { error: `Registration failed: ${error.message}` },
            { status: 400 }
        );
    }
}
