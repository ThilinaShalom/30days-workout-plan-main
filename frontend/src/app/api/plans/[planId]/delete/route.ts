import { NextRequest, NextResponse } from 'next/server';
import { adminDb } from '@/lib/firebase-admin';
import { getSession } from '@/lib/session';

export async function POST(
    request: NextRequest,
    { params }: { params: Promise<{ planId: string }> }
) {
    try {
        const session = await getSession();
        if (!session || session.user_type !== 'customer') {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }

        const { planId } = await params;
        const planRef = adminDb().collection('plans').doc(planId);
        const planDoc = await planRef.get();

        if (!planDoc.exists) {
            return NextResponse.json({ error: 'Plan not found' }, { status: 404 });
        }

        const planData = planDoc.data();
        if (planData?.user_id !== session.user_id) {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }

        await planRef.delete();

        return NextResponse.json({ success: true });
    } catch (error: any) {
        console.error('Error deleting plan:', error);
        return NextResponse.json({ error: error.message }, { status: 500 });
    }
}
