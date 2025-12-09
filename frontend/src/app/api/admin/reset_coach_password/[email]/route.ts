import { NextRequest, NextResponse } from 'next/server';
import { adminAuth } from '@/lib/firebase-admin';
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

        // Generate password reset link
        const resetLink = await adminAuth().generatePasswordResetLink(email);

        // In production, send this via email
        // For now, just return success
        console.log(`Password reset link for ${email}: ${resetLink}`);

        return NextResponse.json({
            success: true,
            message: 'Password reset link generated',
            // Only include link in development
            resetLink: process.env.NODE_ENV === 'development' ? resetLink : undefined,
        });
    } catch (error: any) {
        console.error('Error resetting coach password:', error);
        return NextResponse.json(
            { error: `Failed to reset password: ${error.message}` },
            { status: 500 }
        );
    }
}
