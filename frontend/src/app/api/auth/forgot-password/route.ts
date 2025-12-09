import { NextRequest, NextResponse } from 'next/server';
import { adminAuth } from '@/lib/firebase-admin';

export async function POST(request: NextRequest) {
    try {
        const data = await request.json();
        const { email } = data;

        if (!email) {
            return NextResponse.json(
                { error: 'Email is required' },
                { status: 400 }
            );
        }

        // Check if the email exists in Firebase Authentication
        try {
            await adminAuth().getUserByEmail(email);
        } catch (error) {
            return NextResponse.json(
                { error: 'No account found with this email.' },
                { status: 404 }
            );
        }

        // Generate a password reset link
        const resetLink = await adminAuth().generatePasswordResetLink(email);

        // In production, you would send this via email
        // For now, return it in the response (NOT SECURE - for development only)
        console.log(`Password reset link for ${email}: ${resetLink}`);

        return NextResponse.json({
            success: true,
            message: 'A password reset link has been sent to your email.',
            // Remove this in production - only for development
            resetLink: process.env.NODE_ENV === 'development' ? resetLink : undefined,
        });
    } catch (error: any) {
        console.error('Error processing password reset:', error);
        return NextResponse.json(
            { error: 'Failed to send reset email. Please try again later.' },
            { status: 500 }
        );
    }
}
