import { NextRequest, NextResponse } from 'next/server';
import { adminAuth, adminDb } from '@/lib/firebase-admin';
import { SignJWT } from 'jose';
import { cookies } from 'next/headers';

export async function POST(request: NextRequest) {
    try {
        const data = await request.json();
        const { email, password } = data;

        if (!email || !password) {
            return NextResponse.json(
                { error: 'Email and password are required' },
                { status: 400 }
            );
        }

        // Verify user with Firebase Auth
        let userRecord;
        try {
            userRecord = await adminAuth().getUserByEmail(email);
        } catch (error) {
            return NextResponse.json({ error: 'Invalid credentials' }, { status: 401 });
        }

        // Check if user is a coach (in coaches collection) or customer (in users collection)
        let userData;
        let userType = 'customer';

        // First check coaches collection
        const coachDoc = await adminDb().collection('coaches').doc(userRecord.uid).get();
        if (coachDoc.exists) {
            userData = coachDoc.data();
            userType = 'coach';
        } else {
            // If not in coaches, check users collection
            const userDoc = await adminDb().collection('users').doc(userRecord.uid).get();
            if (!userDoc.exists) {
                return NextResponse.json({ error: 'User not found' }, { status: 401 });
            }
            userData = userDoc.data();
            userType = userData?.user_type || 'customer';
        }

        // Create JWT token - MUST USE SAME SECRET as session.ts
        const secret = new TextEncoder().encode(
            process.env.SESSION_SECRET || 'your-secret-key-change-in-production'
        );

        const token = await new SignJWT({
            user_id: userRecord.uid,
            user_type: userType,
            email: userRecord.email,
            username: userData?.user_name || userData?.username,
        })
            .setProtectedHeader({ alg: 'HS256' })
            .setExpirationTime('24h')
            .sign(secret);

        // Set cookie
        const cookieStore = await cookies();
        cookieStore.set('session', token, {
            httpOnly: true,
            secure: process.env.NODE_ENV === 'production',
            sameSite: 'lax',
            maxAge: 60 * 60 * 24, // 24 hours
            path: '/',
        });

        const redirect = userType === 'customer'
            ? '/dashboard'
            : '/coach/dashboard';

        return NextResponse.json({
            success: true,
            user_type: userType,
            user_name: userData?.user_name || userData?.username,
            redirect,
        });
    } catch (error: any) {
        console.error('Login error:', error);
        return NextResponse.json({ error: 'Login failed' }, { status: 500 });
    }
}
