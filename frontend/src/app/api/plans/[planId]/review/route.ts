import { NextRequest, NextResponse } from 'next/server';
import { adminDb } from '@/lib/firebase-admin';
import { getSession } from '@/lib/session';

export async function POST(
    request: NextRequest,
    { params }: { params: Promise<{ planId: string }> }
) {
    try {
        const session = await getSession();
        if (!session || session.user_type !== 'coach') {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }

        const { planId } = await params;
        const formData = await request.formData();
        const coachComment = formData.get('coach_comment') as string;
        const action = formData.get('action') as string;

        if (!coachComment || !action) {
            return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
        }

        const planRef = adminDb().collection('plans').doc(planId);
        const planDoc = await planRef.get();

        if (!planDoc.exists) {
            return NextResponse.json({ error: 'Plan not found' }, { status: 404 });
        }

        const newStatus = action === 'approve' ? 'approved' : 'rejected';
        await planRef.update({
            coach_comment: coachComment,
            status: newStatus,
            updated_at: new Date().toISOString(),
        });

        return NextResponse.json({ success: true, status: newStatus });
    } catch (error: any) {
        console.error('Error reviewing plan:', error);
        return NextResponse.json({ error: error.message }, { status: 500 });
    }
}
