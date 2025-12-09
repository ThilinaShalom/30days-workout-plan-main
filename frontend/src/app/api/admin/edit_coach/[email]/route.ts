import { NextRequest, NextResponse } from 'next/server';
import { adminAuth, adminDb } from '@/lib/firebase-admin';
import { getSession } from '@/lib/session';

export async function POST(
    request: NextRequest,
    { params }: { params: Promise<{ email: string }> }
) {
    try {
        const session = await getSession();
        if (!session || session.user_type !== 'admin') {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }

        const { email } = await params;
        const data = await request.json();
        const { coach_name, specialization, profile_pic_url, services } = data;

        // Get user by email
        const userRecord = await adminAuth().getUserByEmail(email);

        // Update COACHES collection in Firestore
        await adminDb().collection('coaches').doc(userRecord.uid).update({
            user_name: coach_name,
            username: coach_name,
            specialization: specialization || '',
            profile_pic_url: profile_pic_url || '',
            services: Array.isArray(services) ? services : [],
            updated_at: new Date().toISOString(),
        });

        // Update Firebase Auth display name
        await adminAuth().updateUser(userRecord.uid, {
            displayName: coach_name,
        });

        return NextResponse.json({
            success: true,
            message: 'Coach updated successfully',
        });
    } catch (error: any) {
        console.error('Error editing coach:', error);
        return NextResponse.json(
            { error: `Failed to edit coach: ${error.message}` },
            { status: 500 }
        );
    }
}
