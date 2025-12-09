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

        // Get user by email
        const userRecord = await adminAuth().getUserByEmail(email);

        // Delete from COACHES collection in Firestore
        await adminDb().collection('coaches').doc(userRecord.uid).delete();

        // Delete from Firebase Auth
        await adminAuth().deleteUser(userRecord.uid);

        return NextResponse.json({
            success: true,
            message: 'Coach deleted successfully',
        });
    } catch (error: any) {
        console.error('Error deleting coach:', error);
        return NextResponse.json(
            { error: `Failed to delete coach: ${error.message}` },
            { status: 500 }
        );
    }
}
