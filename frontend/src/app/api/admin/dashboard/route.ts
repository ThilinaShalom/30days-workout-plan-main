import { NextRequest, NextResponse } from 'next/server';
import { adminDb } from '@/lib/firebase-admin';
import { getSession } from '@/lib/session';

export async function GET(request: NextRequest) {
    try {
        console.log('Admin dashboard: Getting session...');
        const session = await getSession();
        console.log('Admin dashboard: Session:', session);

        if (!session) {
            console.log('Admin dashboard: No session found');
            return NextResponse.json({ error: 'No session' }, { status: 401 });
        }

        if (session.user_type !== 'admin') {
            console.log('Admin dashboard: User is not admin:', session.user_type);
            return NextResponse.json({ error: 'Not an admin' }, { status: 401 });
        }

        console.log('Admin dashboard: Fetching coaches...');
        // Get all coaches from Firestore
        const coachesSnapshot = await adminDb()
            .collection('coaches')
            
            .get();

        const coaches = coachesSnapshot.docs.map((doc) => ({
            id: doc.id,
            ...doc.data(),
        }));

        console.log('Admin dashboard: Found', coaches.length, 'coaches');
        return NextResponse.json({ coaches });
    } catch (error: any) {
        console.error('Error fetching admin dashboard:', error);
        return NextResponse.json({ error: error.message }, { status: 500 });
    }
}
