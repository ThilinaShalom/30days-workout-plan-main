import { NextRequest, NextResponse } from 'next/server';
import { adminDb } from '@/lib/firebase-admin';
import { getSession } from '@/lib/session';

export async function GET(request: NextRequest) {
    try {
        const session = await getSession();
        if (!session) {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }

        // Get user data
        const userDoc = await adminDb().collection('users').doc(session.user_id).get();
        const userData = userDoc.data();

        // Get plans for this user
        const plansSnapshot = await adminDb()
            .collection('plans')
            .where('user_id', '==', session.user_id)
            .get();

        const plans = plansSnapshot.docs.map((doc) => ({
            id: doc.id,
            ...doc.data(),
            status: doc.data()?.status || 'not_sent',
        }));

        return NextResponse.json({
            user_name: userData?.user_name || 'Customer',
            plans,
        });
    } catch (error: any) {
        console.error('Error fetching customer dashboard:', error);
        return NextResponse.json({ error: error.message }, { status: 500 });
    }
}
