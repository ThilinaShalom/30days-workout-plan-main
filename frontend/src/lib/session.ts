import { NextRequest } from 'next/server';
import { cookies } from 'next/headers';
import { jwtVerify } from 'jose';

const SECRET = new TextEncoder().encode(process.env.SESSION_SECRET || 'your-secret-key');
export interface SessionData {
    user_id: string;
    user_type: string;
    email?: string;
    username?: string;
}

export async function getSession() {
    try {
        const cookieStore = await cookies();
        const sessionCookie = cookieStore.get('session');

        if (!sessionCookie) {
            console.log('No session cookie found');
            return null;
        }

        const secret = new TextEncoder().encode(
            process.env.SESSION_SECRET || 'your-secret-key-change-in-production'
        );

        try {
            const { payload } = await jwtVerify(sessionCookie.value, secret);
            console.log('Session payload:', payload);

            return {
                user_id: payload.user_id as string,
                user_type: payload.user_type as string,
                email: payload.email as string | undefined,
                username: payload.username as string | undefined,
            };
        } catch (error) {
            console.error('JWT verification failed:', error);
            return null;
        }
    } catch (error) {
        console.error('Error getting session:', error);
        return null;
    }
}
