import * as admin from 'firebase-admin';

// Initialize Firebase Admin SDK (singleton pattern)
const initFirebaseAdmin = () => {
    if (!admin.apps.length) {
        // Check if running in Node.js environment (API routes)
        if (typeof window === 'undefined') {
            try {
                const serviceAccount = require('../hdproject-6e51c-firebase-adminsdk-4e5te-d7102a3fe3.json');

                admin.initializeApp({
                    credential: admin.credential.cert(serviceAccount),
                });

                console.log('Firebase Admin initialized successfully');
            } catch (error) {
                console.error('Error initializing Firebase Admin:', error);
                throw error;
            }
        }
    }
    return admin;
};

export const adminAuth = () => {
    const adminInstance = initFirebaseAdmin();
    return adminInstance.auth();
};

export const adminDb = () => {
    const adminInstance = initFirebaseAdmin();
    return adminInstance.firestore();
};

export default initFirebaseAdmin;
